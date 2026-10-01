import argparse
import gc, os, re, sys
import numpy as np
import pandas as pd
import spacy
import torch
from rapidfuzz import fuzz
from spacy.lang.en.stop_words import STOP_WORDS
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline


class _ListDataset(Dataset):
    """Minimal wrapper so a plain Python list can be fed to a HF pipeline as
    a streaming iterable with num_workers>0 (lets tokenization of the next
    batch overlap with GPU inference on the current one)."""

    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from preprocess.clean_text import clean_text
from sklearn.preprocessing import LabelEncoder

sep = "=" * 80

ASPECT_MODEL_ID = "yangheng/deberta-v3-base-end2end-absa"
SENTI_MODEL_ID = "yangheng/deberta-v3-base-absa-v1.1"

# DeBERTa-v3-base max sequence length; set explicitly because some
# checkpoints ship a sentinel model_max_length and truncation is then skipped.
DEFAULT_MAX_LENGTH = 512

POLARITIES = ("positive", "negative", "neutral")
JUNK_TOKENS = STOP_WORDS

_classifier_error_shown = False
_classifier_error_count = 0
_extractor_error_shown = False
_extractor_error_count = 0


# ---------------------------------------------------------------------------
# Aspect-level noise filtering
# ---------------------------------------------------------------------------

def is_junk_aspect(term: str) -> bool:
    """Bare stopwords, pure numbers, single/double-char fragments."""
    t = term.strip().lower()
    if not t:
        return True
    if t in JUNK_TOKENS:
        return True
    if t.replace(".", "").replace(",", "").replace("-", "").isdigit():
        return True
    if len(t) <= 2:
        return True
    return False


def is_named_entity(term: str, nlp) -> bool:
    """True if the span is (almost) entirely covered by a spaCy named entity."""
    doc = nlp(term)
    if not doc.ents:
        return False
    covered = sum(len(ent.text) for ent in doc.ents)
    return covered >= 0.8 * len(term)


def normalize_sentiment_label(raw_label: str) -> str:
    """Strip tagging-scheme prefixes ("asp-", "aspect-", "b-", "i-") so labels
    such as "ASP-Neutral" or "Positive" normalize to plain lowercase values."""
    label = raw_label.strip().lower()
    for prefix in ("asp-", "aspect-", "b-", "i-"):
        if label.startswith(prefix):
            label = label[len(prefix):]
            break
    return label


# ---------------------------------------------------------------------------
# Sentence segmentation
# ---------------------------------------------------------------------------

def segment_sentences(reviews: list, nlp, n_process: int = 1, batch_size: int = 256) -> list:
    """Sentence-split with nlp.pipe, keeping only the components sentence
    boundaries depend on (the shared nlp object is not modified)."""
    keep = {"tok2vec", "parser", "senter", "sentencizer"}
    disable = [name for name in nlp.pipe_names if name not in keep]

    result = []
    for doc in nlp.pipe(reviews, batch_size=batch_size, n_process=n_process, disable=disable):
        sents = [s.text.strip() for s in doc.sents if s.text.strip()]
        result.append(sents if sents else ([doc.text] if doc.text.strip() else []))
    return result


# ---------------------------------------------------------------------------
# Stage 1: aspect SPAN extraction.
# The end2end extractor is used only to locate aspect spans. Its label/score
# are kept for diagnostics, but polarity always comes from the dedicated
# classifier (Stage 2): depending on the checkpoint's label set, the
# extractor's score may measure span confidence rather than polarity.
# ---------------------------------------------------------------------------

def _clean_aspect(s: str) -> str:
    """Strip leading/trailing punctuation and whitespace from an aspect span."""
    if not s:
        return ""
    return re.sub(r'^[\s.,;:!?()\[\]{}"\']+|[\s.,;:!?()\[\]{}"\']+$', "", s).strip()


def _locate_span(text: str, phrase: str):
    """Locate a phrase's character offsets in text (case-sensitive first)."""
    if not phrase:
        return None
    pat = re.escape(phrase)
    m = re.search(pat, text)
    if m:
        return m.start(), m.end()
    m = re.search(pat, text, flags=re.IGNORECASE)
    if m:
        return m.start(), m.end()
    return None


def extract_aspects_from_entities(text: str, ents: list) -> list:
    """Turn one sentence's raw extractor output into cleaned aspect spans."""
    if not text.strip() or not ents:
        return []

    aspects = []
    seen = set()
    for ent in ents:
        label = (ent.get("entity_group") or "").lower()
        if not any(keyword in label for keyword in ["asp", "aspect", "b-", "i-"]):
            continue

        word = _clean_aspect(ent.get("word", ""))
        if not word:
            continue

        start = ent.get("start")
        end = ent.get("end")
        if start is None or end is None:
            loc = _locate_span(text, word)
            if loc:
                start, end = loc
            else:
                continue

        key = (word.lower(), int(start), int(end))
        if key in seen:
            continue
        seen.add(key)

        aspects.append({
            "aspect": word,
            "start": int(start),
            "end": int(end),
            "extractor_label": ent.get("entity_group", ""),
            "extractor_score": float(ent.get("score", 0.0)),
            # Filled by Stage 2. Probabilities stay NaN unless a real
            # classifier softmax is available (no synthetic distributions).
            "sentiment": None,
            "confidence": np.nan,
            "source": "unclassified",
            "prob_positive": np.nan,
            "prob_negative": np.nan,
            "prob_neutral": np.nan,
        })

    return aspects


def run_aspect_extractor_batched(sentences: list, aspect_extractor, batch_size: int,
                                 chunk_size: int, num_workers: int = 0) -> list:
    """Length-sorted batched extraction; results are returned aligned 1:1
    with the input order."""
    global _extractor_error_shown, _extractor_error_count
    order = sorted(range(len(sentences)), key=lambda i: len(sentences[i]))
    ents_by_original_index = [None] * len(sentences)

    pbar = tqdm(total=len(sentences), desc="Aspect extraction (batched, length-sorted)")
    for i in range(0, len(order), chunk_size):
        idx_chunk = order[i:i + chunk_size]
        text_chunk = [sentences[j] for j in idx_chunk]
        try:
            out = aspect_extractor(_ListDataset(text_chunk), batch_size=batch_size, num_workers=num_workers)
            batch_ents = list(out)
        except Exception as e:
            _extractor_error_count += len(text_chunk)
            if not _extractor_error_shown:
                print(f"  ⚠ Extractor batch call failed (showing first occurrence only; "
                      f"further failures counted silently): {type(e).__name__}: {e}")
                _extractor_error_shown = True
            batch_ents = [[] for _ in text_chunk]

        for j, ents in zip(idx_chunk, batch_ents):
            ents_by_original_index[j] = ents
        pbar.update(len(idx_chunk))
    pbar.close()
    return ents_by_original_index


# ---------------------------------------------------------------------------
# Stage 2: polarity for EVERY aspect via the dedicated (sentence, aspect)
# classifier.
# ---------------------------------------------------------------------------

def apply_classifier_result(aspect_dict: dict, raw_result) -> None:
    """Write the classifier's full softmax into aspect_dict. The classifier
    result always wins; a failed/malformed call leaves the aspect
    unclassified (handled later by apply_extractor_backup)."""
    if not raw_result:
        return
    result = raw_result
    if isinstance(result, list) and len(result) > 0 and isinstance(result[0], list):
        result = result[0]
    if not isinstance(result, list):
        return

    scores = {normalize_sentiment_label(d.get("label", "")): float(d.get("score", 0.0))
              for d in result if isinstance(d, dict)}
    scores = {k: v for k, v in scores.items() if k in POLARITIES}
    if not scores:
        return

    best_label = max(scores, key=scores.get)
    aspect_dict["sentiment"] = best_label
    aspect_dict["confidence"] = scores[best_label]
    aspect_dict["source"] = "classifier"
    aspect_dict["prob_positive"] = scores.get("positive", 0.0)
    aspect_dict["prob_negative"] = scores.get("negative", 0.0)
    aspect_dict["prob_neutral"] = scores.get("neutral", 0.0)


def apply_extractor_backup(aspects: list) -> None:
    """Only for aspects the classifier failed on: keep the extractor's label
    IF it is a genuine polarity label. Probabilities stay NaN, so these rows
    carry a hard label but never a fabricated distribution. Aspects whose
    extractor label is not a polarity stay unclassified and are dropped."""
    for asp in aspects:
        if asp["source"] != "unclassified":
            continue
        parsed = normalize_sentiment_label(asp.get("extractor_label", ""))
        if parsed in POLARITIES:
            asp["sentiment"] = parsed
            asp["confidence"] = asp.get("extractor_score", np.nan)
            asp["source"] = "extractor_backup"


def build_classifier_contexts(sentences_per_review: list, window: int = 1,
                              full_review_max_chars: int = 1500) -> list:
    """Um contexto por sentença, na mesma ordem de flat_sentences.
    - review curto (<= full_review_max_chars): o review inteiro
    - review longo: a sentença +/- `window` sentenças do mesmo review
    - window < 0: sempre o review inteiro (truncado pelo tokenizer em 512 tokens)
    """
    contexts = []
    for sents in sentences_per_review:
        full = " ".join(sents)
        use_full = window < 0 or len(full) <= full_review_max_chars
        for i in range(len(sents)):
            if use_full:
                contexts.append(full)
            else:
                lo, hi = max(0, i - window), min(len(sents), i + window + 1)
                contexts.append(" ".join(sents[lo:hi]))
    return contexts


def run_classifier_batched(all_aspects_per_sentence: list, flat_sentences: list,
                           sent_classifier, batch_size: int, chunk_size: int,
                           num_workers: int = 0) -> None:
    """Classify every (sentence, aspect) pair of the corpus in large,
    length-sorted batches. Mutates all_aspects_per_sentence in place."""
    global _classifier_error_shown, _classifier_error_count

    pairs, locations = [], []
    for si, aspects in enumerate(all_aspects_per_sentence):
        for aj, asp in enumerate(aspects):
            pairs.append({"text": flat_sentences[si], "text_pair": asp["aspect"]})
            locations.append((si, aj))

    print(f"  Classifying polarity for {len(pairs):,} aspect mention(s)")
    if not pairs:
        return

    order = sorted(range(len(pairs)),
                   key=lambda k: len(pairs[k]["text"]) + len(pairs[k]["text_pair"]))

    pbar = tqdm(total=len(order), desc="Sentiment classification (batched, length-sorted)")
    for i in range(0, len(order), chunk_size):
        idx_chunk = order[i:i + chunk_size]
        pair_chunk = [pairs[k] for k in idx_chunk]
        try:
            batch_out = list(sent_classifier(_ListDataset(pair_chunk), batch_size=batch_size,
                                             num_workers=num_workers))
        except Exception as e:
            _classifier_error_count += len(pair_chunk)
            if not _classifier_error_shown:
                print(f"  ⚠ Classifier batch call failed (showing first occurrence only; "
                      f"further failures counted silently): {type(e).__name__}: {e}")
                _classifier_error_shown = True
            batch_out = [None] * len(pair_chunk)

        for k, result in zip(idx_chunk, batch_out):
            si, aj = locations[k]
            apply_classifier_result(all_aspects_per_sentence[si][aj], result)
        pbar.update(len(idx_chunk))
    pbar.close()


# ---------------------------------------------------------------------------
# Full extraction pipeline
# ---------------------------------------------------------------------------

def extract_absa_records(reviews: list, aspect_extractor, sent_classifier, nlp,
                         extract_batch_size: int = 64, classify_batch_size: int = 64,
                         chunk_size: int = 4000, spacy_n_process: int = 1,
                         pipeline_num_workers: int = 0, context_window: int = 1,
                         full_review_max_chars: int = 1500) -> pd.DataFrame:
    print("  Segmenting reviews into sentences...")
    sentences_per_review = segment_sentences(reviews, nlp, n_process=spacy_n_process)
    total_sentences = sum(len(s) for s in sentences_per_review)
    print(f"  {len(reviews):,} reviews -> {total_sentences:,} sentences")

    # Flatten review -> sentences, tracking review_idx in parallel
    flat_sentences, flat_review_idx = [], []
    for ridx, sents in enumerate(sentences_per_review):
        for s in sents:
            flat_sentences.append(s)
            flat_review_idx.append(ridx)

    # One classifier context per sentence, aligned 1:1 with flat_sentences
    flat_contexts = build_classifier_contexts(
        sentences_per_review, window=context_window,
        full_review_max_chars=full_review_max_chars,
    )
    assert len(flat_contexts) == len(flat_sentences), "contexts misaligned with sentences"

    # ---- Stage 1: spans (sentence level) ----
    ents_per_sentence = run_aspect_extractor_batched(
        flat_sentences, aspect_extractor, batch_size=extract_batch_size,
        chunk_size=chunk_size, num_workers=pipeline_num_workers,
    )
    all_aspects_per_sentence = [
        extract_aspects_from_entities(text, ents)
        for text, ents in zip(flat_sentences, ents_per_sentence)
    ]

    # ---- Stage 2: polarity for every aspect (widened context) ----
    run_classifier_batched(
        all_aspects_per_sentence, flat_contexts, sent_classifier,
        batch_size=classify_batch_size, chunk_size=chunk_size,
        num_workers=pipeline_num_workers,
    )
    for aspects in all_aspects_per_sentence:
        apply_extractor_backup(aspects)

    # ---- Flatten (columnar) ----
    cols = {k: [] for k in ("review_idx", "aspect", "sentiment", "score", "source",
                            "extractor_label", "extractor_score",
                            "prob_positive", "prob_negative", "prob_neutral")}
    dropped_unclassified = 0
    for si, aspects in enumerate(all_aspects_per_sentence):
        ridx = flat_review_idx[si]
        for a in aspects:
            if a["source"] == "unclassified":
                dropped_unclassified += 1
                continue
            cols["review_idx"].append(ridx)
            cols["aspect"].append(a["aspect"].lower())
            cols["sentiment"].append(a["sentiment"])
            cols["score"].append(a["confidence"])
            cols["source"].append(a["source"])
            cols["extractor_label"].append(a["extractor_label"])
            cols["extractor_score"].append(a["extractor_score"])
            cols["prob_positive"].append(a["prob_positive"])
            cols["prob_negative"].append(a["prob_negative"])
            cols["prob_neutral"].append(a["prob_neutral"])

    if _extractor_error_count:
        print(f"  ⚠ Extractor call failed for {_extractor_error_count:,} sentence(s) — "
              f"those sentences yielded no aspects.")
    if _classifier_error_count:
        print(f"  ⚠ Classifier call failed for {_classifier_error_count:,} aspect(s) — "
              f"extractor polarity used where available.")
    if dropped_unclassified:
        print(f"  Dropped {dropped_unclassified:,} aspect(s) with no usable polarity.")

    return pd.DataFrame(cols)


def print_polarity_diagnostics(aspect_df: pd.DataFrame, extractor_id2label: dict) -> None:
    """Quick look at label sets and polarity distribution."""
    print(f"  Extractor id2label: {extractor_id2label}")
    if aspect_df.empty:
        return
    print("  Source distribution:")
    print(aspect_df["source"].value_counts(normalize=True).round(3).to_string())
    print("  Sentiment distribution by source:")
    print(aspect_df.groupby("source")["sentiment"].value_counts(normalize=True).round(3).to_string())
    agree = aspect_df.loc[aspect_df["source"] == "classifier"].copy()
    agree["extractor_polarity"] = agree["extractor_label"].map(normalize_sentiment_label)
    agree = agree[agree["extractor_polarity"].isin(POLARITIES)]
    if not agree.empty:
        rate = (agree["extractor_polarity"] == agree["sentiment"]).mean()
        print(f"  Extractor/classifier polarity agreement: {rate:.3f} (n={len(agree):,})")


# ---------------------------------------------------------------------------
# Vocabulary normalization: lemmatization -> typo merge -> frequency filter
# ---------------------------------------------------------------------------

def lemmatize_aspects(aspect_df: pd.DataFrame, nlp, batch_size: int = 1000) -> pd.DataFrame:
    """Map each aspect term to its lemma form ("batteries" -> "battery"),
    so plural/inflectional variants collapse before the fuzzy typo merge
    (fuzz.ratio alone misses them, e.g. battery/batteries ≈ 75)."""
    terms = aspect_df["aspect"].unique().tolist()
    disable = [name for name in nlp.pipe_names if name in ("parser", "ner")]

    lemma_map = {}
    for t, doc in tqdm(zip(terms, nlp.pipe(terms, batch_size=batch_size, disable=disable)),
                       total=len(terms), desc="Lemmatizing aspects"):
        lemma = " ".join(tok.lemma_.lower() for tok in doc if not tok.is_punct).strip()
        lemma_map[t] = lemma if lemma else t

    aspect_df = aspect_df.copy()
    aspect_df["aspect"] = aspect_df["aspect"].map(lemma_map)
    n_after = aspect_df["aspect"].nunique()
    print(f"  {len(terms):,} unique terms -> {n_after:,} after lemmatization")

    before = len(aspect_df)
    aspect_df = aspect_df[~aspect_df["aspect"].apply(is_junk_aspect)].reset_index(drop=True)
    if before - len(aspect_df):
        print(f"  Dropped {before - len(aspect_df):,} mention(s) that became junk after lemmatization")
    return aspect_df


def merge_typo_variants(aspect_df: pd.DataFrame, min_ratio: float = 90.0) -> tuple:
    unique_terms = sorted(aspect_df["aspect"].unique().tolist())
    parent = {t: t for t in unique_terms}

    def find(t):
        while parent[t] != t:
            parent[t] = parent[parent[t]]
            t = parent[t]
        return t

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    buckets = {}
    for t in unique_terms:
        if t:
            buckets.setdefault(t[0], []).append(t)

    for bucket_terms in buckets.values():
        for i, a in enumerate(bucket_terms):
            for b in bucket_terms[i + 1:]:
                if abs(len(a) - len(b)) > 3:
                    continue
                if fuzz.ratio(a, b) >= min_ratio:
                    union(a, b)

    term_counts = aspect_df["aspect"].value_counts().to_dict()
    groups = {}
    for t in unique_terms:
        groups.setdefault(find(t), []).append(t)

    term_to_canonical, merges = {}, {}
    for members in groups.values():
        canonical = max(members, key=lambda t: term_counts.get(t, 0))
        for t in members:
            term_to_canonical[t] = canonical
            if t != canonical:
                merges[t] = canonical

    aspect_df = aspect_df.copy()
    aspect_df["aspect"] = aspect_df["aspect"].map(term_to_canonical)
    print(f"  Merged {len(merges):,} typo variant(s) into canonical spellings")
    return aspect_df, merges


def filter_by_frequency(aspect_df: pd.DataFrame, df: pd.DataFrame, item_col: str,
                        min_reviews: int, min_items: int,
                        split_col: str = None, train_value: str = None) -> tuple:
    """Keep only aspects supported by at least `min_reviews` distinct reviews
    and `min_items` distinct items. This replaces the old cluster-quality
    filter, which re-applied the is_junk_aspect criteria to 1:1 clusters and
    therefore never dropped anything.

    If split_col/train_value are given, support is counted on TRAINING
    reviews only, so the vocabulary does not leak information from the test
    set. Mentions of in-vocabulary aspects are kept for all splits."""
    ref = aspect_df.merge(df[[item_col]], left_on="review_idx", right_index=True, how="left")
    if split_col is not None:
        train_reviews = df.index[df[split_col].astype(str) == str(train_value)]
        ref = ref[ref["review_idx"].isin(train_reviews)]
        print(f"  Counting aspect support on training reviews only "
              f"({len(train_reviews):,} reviews where {split_col} == {train_value!r})")

    stats = (ref.groupby("aspect")
             .agg(n_reviews=("review_idx", "nunique"),
                  n_items=(item_col, "nunique"),
                  n_mentions=("review_idx", "size"))
             .sort_values("n_mentions", ascending=False))
    vocab = stats[(stats["n_reviews"] >= min_reviews) & (stats["n_items"] >= min_items)]

    before_terms = aspect_df["aspect"].nunique()
    before_rows = len(aspect_df)
    aspect_df = aspect_df[aspect_df["aspect"].isin(vocab.index)].reset_index(drop=True)
    print(f"  Vocabulary: {before_terms:,} -> {len(vocab):,} aspects "
          f"(min_reviews={min_reviews}, min_items={min_items})")
    print(f"  Mentions kept: {len(aspect_df):,}/{before_rows:,} "
          f"({100 * len(aspect_df) / max(before_rows, 1):.1f}%)")
    return aspect_df, vocab.reset_index()


def assign_cluster_ids(aspect_df: pd.DataFrame) -> tuple:
    """1:1 term -> id mapping over the cleaned vocabulary. Semantic
    (embedding-based) canonicalization is meant to be applied on top of this
    vocabulary as a later step."""
    aspect_df = aspect_df.copy()
    unique_terms = sorted(aspect_df["aspect"].unique().tolist())
    term_to_cluster = {t: cid for cid, t in enumerate(unique_terms)}
    aspect_df["cluster"] = aspect_df["aspect"].map(term_to_cluster)
    print(f"  {len(unique_terms):,} aspect ids assigned")
    return aspect_df, term_to_cluster


# ---------------------------------------------------------------------------
# Aggregation per (review, aspect)
# ---------------------------------------------------------------------------

def aggregate_absa_results(aspect_df: pd.DataFrame, df: pd.DataFrame,
                           id_cols=("userId", "itemId", "raw_user", "raw_item")) -> pd.DataFrame:
    """One row per (review_idx, cluster).

    - prob_*: mean of the classifier's real softmax (NaN-skipping; rows from
      the extractor backup have NaN probabilities and do not contribute).
    - polarity: prob_positive - prob_negative, in [-1, 1]; neutral mentions
      sit near 0 while still counting as a mention.
    - sentiment: argmax of the mean probabilities when available, else the
      majority hard label (extractor-backup-only groups).
    """
    keys = ["review_idx", "cluster"]
    g = aspect_df.groupby(keys)

    agg = g.agg(
        aspect=("aspect", "first"),
        mentions=("aspect", "size"),
        confidence=("score", "mean"),
        pct_classifier=("source", lambda s: (s == "classifier").mean()),
        prob_positive=("prob_positive", "mean"),
        prob_negative=("prob_negative", "mean"),
        prob_neutral=("prob_neutral", "mean"),
    )

    majority = (aspect_df.groupby(keys + ["sentiment"]).size()
                .unstack(fill_value=0).idxmax(axis=1))
    probs = agg[["prob_positive", "prob_negative", "prob_neutral"]]
    has_probs = probs.notna().all(axis=1)
    soft = probs[has_probs].idxmax(axis=1).str.replace("prob_", "", regex=False)

    agg["sentiment"] = majority
    agg.loc[has_probs, "sentiment"] = soft
    agg["polarity"] = agg["prob_positive"] - agg["prob_negative"]
    agg = agg.reset_index()

    present = [c for c in id_cols if c in df.columns]
    if present:
        agg = agg.merge(df[present], left_on="review_idx", right_index=True, how="left")

    ordered = (["review_idx"] + present +
               ["cluster", "aspect", "sentiment", "polarity", "mentions", "confidence",
                "pct_classifier", "prob_positive", "prob_negative", "prob_neutral"])
    return agg[ordered]


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def load_dataset(args):
    print(sep)
    print(f"LOADING DATASET: {args.dataset_name}")
    print(sep)

    if args.from_prepared_csv:
        csv_path = os.path.join("datasets", f"{args.dataset_name}.csv")
        df = pd.read_csv(csv_path)
        df["review"] = df["review"].fillna("")
        args.text_col = "review"
        print(f"  Loaded prepared CSV: {csv_path} ({df.shape[0]:,} rows)")
    else:
        lu = LabelEncoder()
        if args.file_type == "jsonl":
            df = pd.read_json(os.path.join(args.dataset_path, f"{args.dataset_name}.jsonl"), lines=True)
        elif args.file_type == "parquet":
            df = pd.read_parquet(os.path.join(args.dataset_path, f"{args.dataset_name}.parquet"))
        else:
            raise ValueError(f"Unsupported file_type: {args.file_type!r} (expected 'jsonl' or 'parquet')")

        df["raw_user"] = df[args.user_col].astype(str)
        df["raw_item"] = df[args.item_col].astype(str)
        df["userId"] = lu.fit_transform(df[args.user_col])
        df["itemId"] = lu.fit_transform(df[args.item_col])
        df[args.text_col] = df[args.text_col].apply(clean_text)
        print(f"  Total rows: {df.shape[0]:,}")

    df = df.reset_index(drop=True)

    if args.is_sample:
        df = df.sample(n=min(200, df.shape[0]), random_state=42).reset_index(drop=True)
        print(f"  Running on sample of {df.shape[0]} rows")

    return df


def _ensure_max_length(tokenizer, default=DEFAULT_MAX_LENGTH):
    """Clamp sentinel model_max_length values so truncation actually happens."""
    if tokenizer.model_max_length is None or tokenizer.model_max_length > 100_000:
        tokenizer.model_max_length = default
    return tokenizer


def _str2bool(x):
    return x.strip().lower() == "true"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--file_type", type=str, default=None, choices=["jsonl", "parquet"])
    parser.add_argument("--user_col", type=str, default=None)
    parser.add_argument("--text_col", type=str, default=None)
    parser.add_argument("--item_col", type=str, default=None)
    parser.add_argument("--from_prepared_csv", type=_str2bool, default=False,
                        help="Load datasets/{dataset_name}.csv instead of the raw file.")
    parser.add_argument("--is_sample", type=_str2bool, default=False)
    parser.add_argument("--exclude_named_entities", type=_str2bool, default=False,
                        help="Keep False for movie datasets: actor/director names carry preference.")
    parser.add_argument("--lemmatize", type=_str2bool, default=True)
    parser.add_argument("--typo_min_ratio", type=float, default=90.0)
    parser.add_argument("--min_reviews", type=int, default=5,
                        help="Minimum distinct (training) reviews mentioning an aspect.")
    parser.add_argument("--min_items", type=int, default=3,
                        help="Minimum distinct (training) items whose reviews mention an aspect.")
    parser.add_argument("--split_col", type=str, default=None,
                        help="Optional column marking the split; when set, aspect support "
                             "is counted on training rows only (no test leakage).")
    parser.add_argument("--train_value", type=str, default="train",
                        help="Value of --split_col identifying training rows.")
    parser.add_argument("--extract_batch_size", type=int, default=512)
    parser.add_argument("--classify_batch_size", type=int, default=512)
    parser.add_argument("--chunk_size", type=int, default=4000)
    parser.add_argument("--fp16", type=_str2bool, default=True)
    parser.add_argument("--spacy_n_process", type=int, default=40)
    parser.add_argument("--pipeline_num_workers", type=int, default=0)
    parser.add_argument("--context_window", type=int, default=1,
                        help="Sentenças vizinhas de cada lado dadas ao classificador; -1 = review inteiro.")
    parser.add_argument("--full_review_max_chars", type=int, default=1500,
                        help="Reviews até este tamanho são passados inteiros ao classificador.")
    args = parser.parse_args()

    if not args.from_prepared_csv:
        missing = [n for n in ("dataset_path", "file_type", "user_col", "item_col", "text_col")
                   if getattr(args, n) is None]
        if missing:
            parser.error(f"--{missing[0].replace('_', '-')} is required unless --from_prepared_csv=True "
                         f"(missing: {missing})")

    print(sep)
    print("LOADING ABSA MODELS (span extractor + polarity classifier)")
    print(sep)
    device = 0 if torch.cuda.is_available() else -1
    torch_dtype = torch.float16 if (args.fp16 and torch.cuda.is_available()) else None

    tok_asp = _ensure_max_length(AutoTokenizer.from_pretrained(ASPECT_MODEL_ID, use_fast=True))
    mdl_asp = AutoModelForTokenClassification.from_pretrained(ASPECT_MODEL_ID, torch_dtype=torch_dtype)
    aspect_extractor = pipeline(
        task="token-classification",
        model=mdl_asp,
        tokenizer=tok_asp,
        aggregation_strategy="simple",
        device=device,
    )

    tok_senti = _ensure_max_length(AutoTokenizer.from_pretrained(SENTI_MODEL_ID, use_fast=True))
    sent_classifier = pipeline(
        task="text-classification",
        model=SENTI_MODEL_ID,
        tokenizer=tok_senti,
        device=device,
        top_k=None,
        torch_dtype=torch_dtype,
    )

    nlp = spacy.load("en_core_web_sm")
    df = load_dataset(args)
    item_col = "itemId" if "itemId" in df.columns else args.item_col

    if args.split_col is not None and args.split_col not in df.columns:
        parser.error(f"--split_col {args.split_col!r} not found in dataset columns")

    print(sep)
    print("EXTRACTING ASPECTS (spans) AND CLASSIFYING POLARITY (all aspects)")
    print(sep)
    aspect_df = extract_absa_records(
        df[args.text_col].tolist(), aspect_extractor, sent_classifier, nlp,
        extract_batch_size=args.extract_batch_size,
        classify_batch_size=args.classify_batch_size,
        chunk_size=args.chunk_size,
        spacy_n_process=args.spacy_n_process,
        pipeline_num_workers=args.pipeline_num_workers,
        context_window=args.context_window,
        full_review_max_chars=args.full_review_max_chars,
    )

    print(f"  Total aspect mentions: {len(aspect_df):,}")
    print_polarity_diagnostics(aspect_df, mdl_asp.config.id2label)

    print("  Releasing GPU memory...")
    del aspect_extractor, sent_classifier, mdl_asp
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(sep)
    print("FILTERING JUNK ASPECTS")
    print(sep)
    before = len(aspect_df)
    aspect_df = aspect_df[~aspect_df["aspect"].apply(is_junk_aspect)].reset_index(drop=True)
    print(f"  Dropped {before - len(aspect_df):,} junk aspect mention(s)")

    if args.exclude_named_entities:
        before = len(aspect_df)
        ent_terms = {t for t in aspect_df["aspect"].unique() if is_named_entity(t, nlp)}
        aspect_df = aspect_df[~aspect_df["aspect"].isin(ent_terms)].reset_index(drop=True)
        print(f"  Dropped {before - len(aspect_df):,} named-entity aspect mention(s)")

    print(sep)
    print("NORMALIZING VOCABULARY (lemmatize -> typo merge -> frequency filter)")
    print(sep)
    if args.lemmatize:
        aspect_df = lemmatize_aspects(aspect_df, nlp)
    aspect_df, typo_merges = merge_typo_variants(aspect_df, min_ratio=args.typo_min_ratio)
    aspect_df, vocab_stats = filter_by_frequency(
        aspect_df, df, item_col=item_col,
        min_reviews=args.min_reviews, min_items=args.min_items,
        split_col=args.split_col, train_value=args.train_value,
    )
    aspect_df, term_to_cluster = assign_cluster_ids(aspect_df)
    vocab_stats["cluster"] = vocab_stats["aspect"].map(term_to_cluster)

    print(sep)
    print("AGGREGATING PER (REVIEW, ASPECT)")
    print(sep)
    aspect_df_agg = aggregate_absa_results(aspect_df, df, id_cols=("userId", "itemId", "raw_user", "raw_item"))
    print(f"  Aggregated rows: {len(aspect_df_agg):,}")
    print(f"  Reviews with ≥1 aspect: {aspect_df_agg['review_idx'].nunique():,}/{len(df):,}")

    print(sep)
    print(f"SAVING RESULTS FOR: {args.dataset_name}")
    print(sep)
    os.makedirs("results", exist_ok=True)
    agg_path = f"results/{args.dataset_name}_absa_test_aspects.csv"
    vocab_path = f"results/{args.dataset_name}_absa_vocab.csv"
    merges_path = f"results/{args.dataset_name}_absa_typo_merges.csv"

    aspect_df_agg.to_csv(agg_path, index=False)
    vocab_stats.to_csv(vocab_path, index=False)
    pd.DataFrame(sorted(typo_merges.items()), columns=["variant", "canonical"]).to_csv(merges_path, index=False)

    print(f"  Aggregated aspects : {agg_path}")
    print(f"  Vocabulary stats   : {vocab_path}")
    print(f"  Typo merges        : {merges_path}")

    print(sep)
    print("DONE")
    print(sep)


if __name__ == "__main__":
    main()
