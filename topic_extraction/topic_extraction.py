import argparse
import gc
import json
import os
import re
import sys
import time
import warnings

warnings.filterwarnings("ignore", message="The TBB threading layer requires TBB")

warnings.filterwarnings("ignore", message="n_jobs value .* overridden to 1 by setting random_state")

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
from sklearn.preprocessing import LabelEncoder

sep = "=" * 80

EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"


DOMAIN_STOPWORDS = [
    "movie", "movies", "film", "films", "flick", "watch", "watched", "watching",
    "seen", "see", "saw", "just", "like", "really", "think", "thought", "know",
    "lot", "way", "thing", "things", "don", "didn", "doesn", "isn", "wasn",
    "ve", "ll", "re", "br", "dont", "im", "also", "even", "much", "get",
    "got", "make", "made", "say", "said", "going", "one", "two", "would",
    "could", "review", "reviews", "rating", "stars",
]


# ---------------------------------------------------------------------------
# Text cleaning / loading
# ---------------------------------------------------------------------------

def basic_clean(text) -> str:
    """Light cleaning only -- sentence encoders work best on natural text."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    text = str(text)
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)   # IMDb line breaks
    text = re.sub(r"<[^>]+>", " ", text)                           # other HTML
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)            # URLs
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_cleaner(name: str):
    if name == "project":
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.append(os.path.dirname(current_dir))
        from preprocess.clean_text import clean_text  # noqa: E402
        return clean_text
    return basic_clean


def _find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_dataset(args) -> pd.DataFrame:
    print(sep)
    print(f"LOADING DATASET: {args.dataset_name}")
    print(sep)

    lu = LabelEncoder()
    path = os.path.join(args.dataset_path, f"{args.dataset_name}.parquet")
    df = pd.read_parquet(path)
    text_col, user_col, item_col = args.text_col, args.user_col, args.item_col
    print(f"  Loaded raw file: {path} ({len(df):,} rows)")

    df = df.reset_index(drop=True)
    out = pd.DataFrame({"row_id": np.arange(len(df))})
    # Keep the encoded ids that the recommender uses; re-encode only when absent.
    if user_col is not None:
        out["userId"] = df[user_col].values if user_col == "userId" else lu.fit_transform(df[user_col])
    if item_col is not None:
        out["itemId"] = df[item_col].values if item_col == "itemId" else lu.fit_transform(df[item_col])

    cleaner = get_cleaner(args.cleaner)
    out["review"] = df[text_col].apply(cleaner)

    if args.is_sample:
        out = out.sample(n=min(args.sample_size, len(out)), random_state=args.seed).reset_index(drop=True)
        print(f"  Running on a sample of {len(out):,} rows")

    lengths = out["review"].str.split().str.len()
    print(f"  Review length (words): mean={lengths.mean():.1f}, median={lengths.median():.0f}, "
          f"p95={lengths.quantile(0.95):.0f}")
    return out
    

# ---------------------------------------------------------------------------
# Embeddings and dimensionality reduction (computed once, reused across K)
# ---------------------------------------------------------------------------

def load_encoder(device: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBEDDING_MODEL, device=device)


def chunk_docs(docs, chunk_words):
    """Split each doc into consecutive chunks of <= chunk_words words.
    Returns the chunks, the index of the doc each chunk came from (sorted), and word counts."""
    chunks, owner, n_words = [], [], []
    for i, d in enumerate(docs):
        w = d.split()
        if len(w) <= chunk_words:
            chunks.append(d)
            owner.append(i)
            n_words.append(max(1, len(w)))
            continue
        for s in range(0, len(w), chunk_words):
            part = w[s:s + chunk_words]
            chunks.append(" ".join(part))
            owner.append(i)
            n_words.append(len(part))
    return chunks, np.asarray(owner), np.asarray(n_words, dtype=np.float32)


def compute_embeddings(docs, encoder, model_key, args) -> np.ndarray:
    os.makedirs(args.cache_dir, exist_ok=True)
    tag = "sample" if args.is_sample else "full"
    chunk_tag = f"_chunk{args.chunk_words}" if args.chunk_words > 0 else ""
    path = os.path.join(args.cache_dir, f"{args.dataset_name}_{tag}_{model_key}{chunk_tag}_{len(docs)}.npy")
    if os.path.exists(path):
        print(f"  Loading cached embeddings: {path}")
        return np.load(path)

    max_len = encoder.max_seq_length
    n_words = np.array([len(d.split()) for d in docs])
    approx_word_limit = int(max_len * 0.75)  # ~1.3 tokens per English word
    n_long = int((n_words > approx_word_limit).sum())
    print(f"  ~{n_long:,} reviews ({n_long / max(1, len(docs)):.1%}) exceed ~{approx_word_limit} words "
          f"(max_seq_length={max_len} tokens)")

    encode_kw = dict(batch_size=args.embed_batch_size, show_progress_bar=True,
                     normalize_embeddings=True, convert_to_numpy=True)
    if args.chunk_words <= 0:
        if n_long:
            print("  ! Those reviews are truncated; use --chunk_words 200 to embed them in full.")
        print(f"  Encoding {len(docs):,} reviews with {EMBEDDING_MODEL}")
        emb = encoder.encode(docs, **encode_kw).astype(np.float32)
    else:
        chunks, owner, weights = chunk_docs(docs, args.chunk_words)
        print(f"  Encoding {len(docs):,} reviews as {len(chunks):,} chunks of <= {args.chunk_words} "
              f"words with {EMBEDDING_MODEL}")
        emb_c = encoder.encode(chunks, **encode_kw).astype(np.float32)
        # length-weighted mean of the chunk embeddings of each review, then re-normalised
        starts = np.flatnonzero(np.r_[True, owner[1:] != owner[:-1]])
        emb = np.add.reduceat(emb_c * weights[:, None], starts, axis=0)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
        emb = emb.astype(np.float32)
        del emb_c
    np.save(path, emb)
    return emb


def reduce_embeddings(emb: np.ndarray, args) -> np.ndarray:
    if args.use_cuml:
        from cuml.manifold import UMAP
    else:
        from umap import UMAP
    kwargs = {"random_state": args.seed}
    if not args.use_cuml and args.umap_n_jobs != 1:
        # umap-learn only runs multi-threaded without a seed, so results are not reproducible
        kwargs = {"n_jobs": args.umap_n_jobs}
    backend = "cuML (GPU)" if args.use_cuml else f"umap-learn ({kwargs})"
    print(f"  UMAP [{backend}]: n_neighbors={args.umap_neighbors}, "
          f"n_components={args.umap_components}, metric=cosine")
    umap_model = UMAP(n_neighbors=args.umap_neighbors, n_components=args.umap_components,
                      min_dist=0.0, metric="cosine", **kwargs)
    return umap_model.fit_transform(emb)


class PrecomputedReducer:
    """Feeds BERTopic a UMAP projection computed once per embedding model, so the
    K-grid search only re-runs K-Means + c-TF-IDF instead of UMAP every time."""

    def __init__(self, reduced: np.ndarray):
        self.reduced = reduced

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X):
        if X.shape[0] != self.reduced.shape[0]:
            raise ValueError("PrecomputedReducer only supports the documents it was built for.")
        return self.reduced


# ---------------------------------------------------------------------------
# BERTopic model
# ---------------------------------------------------------------------------

def build_vectorizer(args) -> CountVectorizer:
    # NOTE: in BERTopic the vectorizer runs on one concatenated document PER TOPIC,
    # so min_df / max_df count topics, not reviews. Keep min_df=1 and cap the
    # vocabulary with max_features instead.
    return CountVectorizer(
        stop_words=sorted(ENGLISH_STOP_WORDS | set(DOMAIN_STOPWORDS)),
        ngram_range=(1, args.ngram_max),
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
        min_df=1,
        max_features=args.max_features,
    )


def build_topic_model(k, encoder, reduced, args):
    from bertopic import BERTopic
    from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
    from bertopic.vectorizers import ClassTfidfTransformer

    representation = KeyBERTInspired(top_n_words=max(args.top_n_words, args.diversity_top_n),
                                     random_state=args.seed)
    if args.mmr_diversity > 0:
        # removes near-duplicate n-grams such as "special effects" / "effects special"
        representation = [representation,
                          MaximalMarginalRelevance(diversity=args.mmr_diversity,
                                                   top_n_words=max(args.top_n_words, args.diversity_top_n))]

    return BERTopic(
        embedding_model=encoder,
        umap_model=PrecomputedReducer(reduced),
        hdbscan_model=KMeans(n_clusters=k, n_init=10, random_state=args.seed),
        vectorizer_model=build_vectorizer(args),
        ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=True),
        representation_model={"Main": representation},
        top_n_words=max(args.top_n_words, args.diversity_top_n),
        calculate_probabilities=False,
        verbose=False,
    )


# ---------------------------------------------------------------------------
# Evaluation: coherence (C_V, C_NPMI) and topic diversity
# ---------------------------------------------------------------------------

class CoherenceEvaluator:
    """Tokenises a fixed reference sample once with the SAME analyzer used by the
    topic model (so n-gram keywords such as 'special effects' are valid tokens),
    then scores any topic model against it."""

    def __init__(self, docs, args):
        from gensim.corpora import Dictionary

        rng = np.random.RandomState(args.seed)
        n = min(args.coherence_docs, len(docs))
        idx = rng.choice(len(docs), size=n, replace=False)
        analyzer = build_vectorizer(args).build_analyzer()
        self.texts = [analyzer(docs[i]) for i in idx]
        self.texts = [t for t in self.texts if t]
        self.dictionary = Dictionary(self.texts)
        self.top_n = args.coherence_top_n
        self.processes = args.coherence_processes
        print(f"  Coherence reference corpus: {len(self.texts):,} reviews, "
              f"{len(self.dictionary):,} tokens")

    def topic_words(self, topic_model, n):
        words = []
        for tid in sorted(t for t in topic_model.get_topics() if t != -1):
            ws = [w for w, _ in topic_model.get_topic(tid)][:n]
            words.append([w for w in ws if w])
        return words

    def score(self, topic_model, diversity_top_n=25) -> dict:
        from gensim.models.coherencemodel import CoherenceModel

        topics = [[w for w in ws if w in self.dictionary.token2id]
                  for ws in self.topic_words(topic_model, self.top_n)]
        topics = [t for t in topics if len(t) >= 2]
        out = {}
        for measure in ("c_v", "c_npmi"):
            cm = CoherenceModel(topics=topics, texts=self.texts, dictionary=self.dictionary,
                                coherence=measure, processes=self.processes, topn=self.top_n)
            out[measure] = float(cm.get_coherence())

        div_words = self.topic_words(topic_model, diversity_top_n)
        flat = [w for ws in div_words for w in ws]
        out["topic_diversity"] = len(set(flat)) / max(1, len(flat))
        return out


# ---------------------------------------------------------------------------
# Soft topic distributions and user/item profiles
# ---------------------------------------------------------------------------

def soft_topic_distribution(emb, emb_fit, topics_fit, k, temperature):
    """softmax(cos(review, topic centroid) / T). Embeddings are L2-normalised,
    centroids are the mean embedding of each topic's (fitted) reviews."""
    topics_fit = np.asarray(topics_fit)
    centroids = np.stack([emb_fit[topics_fit == t].mean(axis=0) for t in range(k)])
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12
    logits = (emb @ centroids.T) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    return probs.astype(np.float32)


def group_profile(ids, dist, n):
    prof = np.zeros((n, dist.shape[1]), dtype=np.float64)
    cnt = np.zeros(n, dtype=np.float64)
    np.add.at(prof, ids, dist)
    np.add.at(cnt, ids, 1.0)
    prof[cnt > 0] /= cnt[cnt > 0, None]
    return prof.astype(np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def str2bool(x):
    return str(x).strip().lower() in ("true", "1", "yes")


def parse_args():
    p = argparse.ArgumentParser(description="BERTopic topic extraction with automatic model selection")
    # data
    p.add_argument("--dataset_name", required=True)
    p.add_argument("--prepared_dir", default="datasets")
    p.add_argument("--dataset_path", default="data")
    p.add_argument("--file_type", default=None, choices=[None, "jsonl", "parquet", "csv"])
    p.add_argument("--user_col", default='userId')
    p.add_argument("--item_col", default='itemId')
    p.add_argument("--text_col", default='review')
    p.add_argument("--cleaner", default="basic", choices=["basic", "project"],
                   help="basic = HTML/URL/whitespace only (recommended for encoders); "
                        "project = preprocess.clean_text")
    p.add_argument("--min_words", type=int, default=3,
                   help="Reviews with fewer words are not used to FIT topics; they still "
                        "receive a distribution from their embedding.")
    p.add_argument("--is_sample", type=str2bool, default=False)
    p.add_argument("--sample_size", type=int, default=3000)
    p.add_argument("--k_grid", nargs="+", type=int,
                   default=[10, 15, 20, 25, 30, 40, 50, 60, 75, 100],
                   help="Numbers of topics to try (UMAP is computed once, so extra K values "
                        "only cost K-Means + c-TF-IDF + coherence).")
    p.add_argument("--k_range", nargs=3, type=int, default=None, metavar=("START", "STOP", "STEP"),
                   help="Alternative to --k_grid: try K = START, START+STEP, ..., STOP (inclusive). "
                        "E.g. --k_range 10 150 10")
    p.add_argument("--chunk_words", type=int, default=0,
                   help="Split reviews longer than this many words into chunks, embed each chunk "
                        "and average them (weighted by length), so long reviews (e.g. IMDb) are "
                        "not truncated at the encoder's max_seq_length. 0 = off. ~200 is safe "
                        "for all-mpnet-base-v2 (384 tokens).")
    p.add_argument("--selection_metric", default="c_v", choices=["c_v", "c_npmi", "topic_diversity"])
    # model hyper-parameters (shared by all datasets)
    p.add_argument("--umap_neighbors", type=int, default=15)
    p.add_argument("--umap_components", type=int, default=5)
    p.add_argument("--use_cuml", type=str2bool, default=False)
    p.add_argument("--umap_n_jobs", type=int, default=1,
                   help="CPU threads for umap-learn (-1 = all cores). Anything other than 1 "
                        "drops the UMAP seed, so runs are no longer reproducible.")
    p.add_argument("--ngram_max", type=int, default=3)
    p.add_argument("--max_features", type=int, default=100_000)
    p.add_argument("--top_n_words", type=int, default=10)
    p.add_argument("--mmr_diversity", type=float, default=0.3,
                   help="MMR re-ranking after KeyBERTInspired (0 disables).")
    p.add_argument("--temperature", type=float, default=0.05,
                   help="Softmax temperature for the per-review topic distribution.")

    p.add_argument("--coherence_docs", type=int, default=20_000)
    p.add_argument("--coherence_top_n", type=int, default=10)
    p.add_argument("--diversity_top_n", type=int, default=25)
    p.add_argument("--coherence_processes", type=int, default=1)
    p.add_argument("--embed_batch_size", type=int, default=128)
    p.add_argument("--cache_dir", default="results/embeddings")
    p.add_argument("--out_dir", default="results/topics")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_figures", type=str2bool, default=False)
    # cross-validation folds (from utils/leave_one_out_cv.py)
    p.add_argument("--split_file", default=None,
                   help="Fold CSV with row_id/split columns; use {fold} as a placeholder, "
                        "e.g. splits/rotten_tomatoes_fold{fold}.csv. Topics are then fit on "
                        "that fold's train reviews only.")
    p.add_argument("--folds", nargs="+", type=int, default=[0],
                   help="Folds to run when --split_file is given.")
    args = p.parse_args()
    if args.k_range is not None:
        start, stop, step = args.k_range
        if step <= 0 or start < 2 or stop < start:
            p.error("--k_range needs START >= 2, STOP >= START and STEP > 0")
        args.k_grid = list(range(start, stop + 1, step))
    args.k_grid = sorted(set(args.k_grid))
    if args.k_grid[0] < 2:
        p.error("every K must be >= 2")

    return args


def load_train_mask(df: pd.DataFrame, path: str) -> np.ndarray:
    """True for reviews in this fold's train split. Rows are matched by row_id, which is
    the row position in the dataset file in both this script and leave_one_out_cv.py."""
    sp = pd.read_csv(path).set_index("row_id")
    split = df["row_id"].map(sp["split"])
    if split.isna().any():
        raise ValueError(f"{int(split.isna().sum()):,} reviews are missing from {path}; "
                         "was it built from the same dataset file?")
    if "userId" in df and "userId" in sp:
        if not (df["row_id"].map(sp["userId"]).values == df["userId"].values).all():
            raise ValueError(f"userId in {path} does not match the dataset; regenerate the splits.")
    print(f"  Split {path}: {split.value_counts().to_dict()}")
    return (split == "train").values


def run_topics(df, all_docs, emb_all, encoder, long_enough, train_mask, out_dir, args, extra_meta):
    """Search K, fit the best BERTopic model and save topics, distributions and profiles.
    When train_mask is given, only train reviews are used to fit topics and build profiles."""
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    fit_mask = long_enough if train_mask is None else (long_enough & train_mask)
    fit_idx = np.where(fit_mask)[0]
    fit_docs = [all_docs[i] for i in fit_idx]
    print(f"  {len(fit_docs):,}/{len(all_docs):,} reviews used to fit topics "
          f"(>= {args.min_words} words{', train split only' if train_mask is not None else ''})")

    print(sep)
    print("BUILDING COHERENCE REFERENCE CORPUS")
    print(sep)
    evaluator = CoherenceEvaluator(fit_docs, args)

    emb_fit = emb_all[fit_idx]
    reduced = reduce_embeddings(emb_fit, args)

    search_rows = []
    best = None  # dict with model, topics, k, scores
    for k in args.k_grid:
        if k >= len(fit_docs):
            continue
        ts = time.time()
        tm = build_topic_model(k, encoder, reduced, args)
        topics, _ = tm.fit_transform(fit_docs, embeddings=emb_fit)
        scores = evaluator.score(tm, args.diversity_top_n)
        row = {"embedding_model": EMBEDDING_MODEL, "k": k, **scores,
               "min_topic_size": int(np.bincount(topics).min()),
               "fit_seconds": round(time.time() - ts, 1)}
        search_rows.append(row)
        print(f"  K={k:>3}  C_V={scores['c_v']:.4f}  C_NPMI={scores['c_npmi']:.4f}  "
              f"diversity={scores['topic_diversity']:.3f}  ({row['fit_seconds']}s)")

        if best is None or scores[args.selection_metric] > best["scores"][args.selection_metric]:
            best = {"model": tm, "topics": np.asarray(topics), "k": k, "scores": scores}
        else:
            del tm
    del reduced
    gc.collect()

    search_df = pd.DataFrame(search_rows).sort_values(args.selection_metric, ascending=False)
    search_df.to_csv(os.path.join(out_dir, "topic_search.csv"), index=False)

    print(sep)
    print(f"SELECTED: K={best['k']}  (by {args.selection_metric}={best['scores'][args.selection_metric]:.4f})")
    print(sep)

    tm, k = best["model"], best["k"]

    # ---- topic table -------------------------------------------------------
    info = tm.get_topic_info()
    info = info[info["Topic"] != -1].copy()
    info["keywords"] = info["Topic"].apply(lambda t: ", ".join(w for w, _ in tm.get_topic(t)[:args.top_n_words]))
    info["representative_reviews"] = info["Representative_Docs"].apply(
        lambda ds: " ||| ".join(d[:300] for d in (ds or [])[:3]))
    info = info[["Topic", "Count", "keywords", "representative_reviews"]]
    info.to_csv(os.path.join(out_dir, "topic_info.csv"), index=False)
    print(info[["Topic", "Count", "keywords"]].to_string(index=False))

    # ---- per-review distributions for EVERY review (incl. short and val/test ones) --
    # centroids come from the fitted reviews only
    dist = soft_topic_distribution(emb_all, emb_fit, best["topics"], k, args.temperature)
    hard = dist.argmax(axis=1)
    hard[fit_idx] = best["topics"]

    agreement = float((dist[fit_idx].argmax(1) == best["topics"]).mean())
    mean_max_prob = float(dist.max(1).mean())
    print(f"  Soft/hard agreement (argmax == K-Means topic): {agreement:.3f}; "
          f"mean max prob: {mean_max_prob:.3f}")

    np.save(os.path.join(out_dir, "review_topic_dist.npy"), dist)
    review_df = df.drop(columns=["review"]).copy()
    review_df["topic"] = hard
    if train_mask is not None:
        review_df["is_train"] = train_mask
    for t in range(k):
        review_df[f"topic_prob_{t}"] = dist[:, t]
    try:
        review_df.to_parquet(os.path.join(out_dir, "review_topics.parquet"), index=False)
    except Exception:
        review_df.to_csv(os.path.join(out_dir, "review_topics.csv"), index=False)

    # user/item profiles: train reviews only when a fold is given (no val/test leakage)
    prof_mask = np.ones(len(df), dtype=bool) if train_mask is None else train_mask
    if "itemId" in review_df:
        n_items = int(review_df["itemId"].max()) + 1
        np.save(os.path.join(out_dir, "item_topic_profile.npy"),
                group_profile(review_df["itemId"].values[prof_mask].astype(int), dist[prof_mask], n_items))
    if "userId" in review_df:
        n_users = int(review_df["userId"].max()) + 1
        np.save(os.path.join(out_dir, "user_topic_profile.npy"),
                group_profile(review_df["userId"].values[prof_mask].astype(int), dist[prof_mask], n_users))

    # ---- save model + metadata ----------------------------------------------
    try:
        tm.umap_model = None  # precomputed reducer is not reusable on new docs
        tm.topics_ = [int(t) for t in tm.topics_]  # numpy ints break the JSON config
        tm.topic_sizes_ = {int(t): int(c) for t, c in tm.topic_sizes_.items()}
        tm.vectorizer_model.vocabulary_ = {w: int(i) for w, i in tm.vectorizer_model.vocabulary_.items()}
        tm.save(os.path.join(out_dir, "bertopic_model"), serialization="safetensors",
                save_ctfidf=True, save_embedding_model=EMBEDDING_MODEL)
    except Exception as e:
        print(f"  ⚠ Could not save BERTopic model: {type(e).__name__}: {e}")

    if args.save_figures:
        try:
            tm.visualize_barchart(top_n_topics=k, n_words=8).write_html(
                os.path.join(out_dir, "topics_barchart.html"))
            tm.visualize_heatmap().write_html(os.path.join(out_dir, "topics_similarity.html"))
        except Exception as e:
            print(f"  ⚠ Figures skipped: {type(e).__name__}: {e}")

    metrics = {
        "dataset": args.dataset_name,
        **extra_meta,
        "n_reviews": len(all_docs),
        "n_reviews_fit": len(fit_docs),
        "profiles_from": "all reviews" if train_mask is None else "train reviews only",
        "selected_embedding_model": EMBEDDING_MODEL,
        "selected_k": k,
        "selection_metric": args.selection_metric,
        "scores": best["scores"],
        "soft_hard_agreement": agreement,
        "mean_max_topic_prob": mean_max_prob,
        "coherence_definition": {
            "c_v": "Roeder et al. (2015) C_V: NPMI-based indirect cosine confirmation, "
                   "boolean sliding window (110), top-%d words per topic" % args.coherence_top_n,
            "c_npmi": "Average normalised PMI over word pairs, sliding window (10), "
                      "top-%d words per topic" % args.coherence_top_n,
            "topic_diversity": "Unique words / total words in top-%d of all topics "
                               "(Dieng et al., 2020)" % args.diversity_top_n,
            "reference_corpus_reviews": len(evaluator.texts),
        },
        "hyperparameters": {
            "umap": {"n_neighbors": args.umap_neighbors, "n_components": args.umap_components,
                     "min_dist": 0.0, "metric": "cosine"},
            "clustering": "KMeans(n_init=10)",
            "k_grid": args.k_grid,
            "embedding_model": EMBEDDING_MODEL,
            "chunk_words": args.chunk_words,
            "vectorizer": {"ngram_range": [1, args.ngram_max], "max_features": args.max_features,
                           "stopwords": "sklearn English + DOMAIN_STOPWORDS"},
            "ctfidf": "ClassTfidfTransformer(reduce_frequent_words=True)",
            "representation": "KeyBERTInspired -> MaximalMarginalRelevance(diversity=%s)" % args.mmr_diversity,
            "soft_distribution": f"softmax(cos(review, topic centroid) / {args.temperature})",
            "min_words": args.min_words,
            "seed": args.seed,
        },
        "domain_stopwords": DOMAIN_STOPWORDS,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(sep)
    print(f"DONE in {metrics['runtime_seconds']}s -> {out_dir}")
    print(sep)


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_dir = os.path.join(args.out_dir, args.dataset_name)

    df = load_dataset(args)
    all_docs = df["review"].tolist()
    long_enough = (df["review"].str.split().str.len() >= args.min_words).values

    # embeddings are computed (or loaded from cache) once and shared by all folds
    print(sep)
    print(f"EMBEDDING MODEL: {EMBEDDING_MODEL}")
    print(sep)
    encoder = load_encoder(device)
    emb_all = compute_embeddings(all_docs, encoder, EMBEDDING_MODEL.split("/")[-1], args)

    if args.split_file is None:
        run_topics(df, all_docs, emb_all, encoder, long_enough, None, base_dir, args, {})
        return

    for fold in args.folds:
        path = args.split_file.format(fold=fold)
        print(sep)
        print(f"FOLD {fold}")
        print(sep)
        train_mask = load_train_mask(df, path)
        run_topics(df, all_docs, emb_all, encoder, long_enough, train_mask,
                   os.path.join(base_dir, f"fold{fold}"), args, {"fold": fold, "split_file": path})
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()