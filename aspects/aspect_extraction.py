import argparse
import gc
import math
import os
import sys
import tempfile
import numpy as np
import pandas as pd
import spacy
import torch
from spacy.lang.en.stop_words import STOP_WORDS
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from preprocess.clean_text import clean_text
from sklearn.preprocessing import LabelEncoder



NOISE_TOKENS = STOP_WORDS | {
    "it", "that", "this", "what", "i", "you", "we", "they", "thing", "something", "anything", "disc", "dvd", "blu ray"
}

# Generic, low-signal nouns that pass the dep/POS filter but carry almost no
# discriminative topical content (near-universal across every review). Kept
# separate from NOISE_TOKENS so it's easy to tune independently.
GENERIC_ASPECT_NOUNS = {
    "lot", "people", "way", "part", "bit", "kind",
    "sort", "thing", "things", "stuff",
}

# POS tags stripped from the FRONT of a noun chunk so "the story", "this story",
# "a story", "story" all normalize to the same underlying phrase.
DROP_LEADING_POS = {"DET", "PRON"}

ONNX_MODEL_PATH = "models/roberta-sentiment-onnx"


def process_in_chunks(df, chunk_size=20_000, n_process=40):
    chunks = math.ceil(len(df) / chunk_size)
    print(f"  Processing {len(df):,} rows in {chunks} chunks of {chunk_size:,}")

    tmp_files = []
    for i in range(chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, len(df))
        chunk = df.iloc[start:end]

        print(f"\n  Chunk {i + 1}/{chunks} (rows {start:,}–{end:,})")

        reviews = chunk["review"].tolist()
        row_iter = chunk.itertuples(index=False)

        records = []
        for idx, (doc, row) in enumerate(tqdm(
                zip(nlp.pipe(reviews, batch_size=256, n_process=n_process), row_iter),
                total=len(chunk),
                desc=f"  spaCy chunk {i + 1}/{chunks}",
        )):
            for a in extract_aspects_from_doc(doc):
                records.append({
                    "review_idx": start + idx,
                    "review": row.review,
                    "aspect": a["aspect"],
                    "aspect_raw": a["aspect_raw"],
                })

        tmp_path = os.path.join(tempfile.gettempdir(), f"aspects_chunk_{i}.parquet")
        pd.DataFrame(records).to_parquet(tmp_path, index=False)
        tmp_files.append(tmp_path)

        del reviews, records
        gc.collect()

    dfs = []
    for f in tmp_files:
        dfs.append(pd.read_parquet(f))
        os.remove(f)
    result = pd.concat(dfs, ignore_index=True)
    del dfs
    gc.collect()
    return result


def load_sentiment_model_trt(model_name: str, batch_size: int):
    """
    Export the model to ONNX with TensorRT execution provider on first run,
    cache locally. Subsequent runs load from ONNX_MODEL_PATH directly.
    TensorRT kernels are compiled by ONNX Runtime on first inference batch.
    """
    from optimum.onnxruntime import ORTModelForSequenceClassification

    if os.path.exists(ONNX_MODEL_PATH):
        print(f"  Loading ONNX model from cache: {ONNX_MODEL_PATH}")
        _model = ORTModelForSequenceClassification.from_pretrained(
            ONNX_MODEL_PATH,
            provider="TensorrtExecutionProvider",  # TensorRT backend
        )
    else:
        print(f"  Exporting {model_name} to ONNX + TensorRT (runs once)...")
        os.makedirs(ONNX_MODEL_PATH, exist_ok=True)
        _model = ORTModelForSequenceClassification.from_pretrained(
            model_name,
            export=True,
            provider="TensorrtExecutionProvider",
        )
        _model.save_pretrained(ONNX_MODEL_PATH)
        print(f"  ONNX model saved to: {ONNX_MODEL_PATH}")

    _tokenizer = AutoTokenizer.from_pretrained(model_name)

    return pipeline(
        "text-classification",
        model=_model,
        tokenizer=_tokenizer,
        device=0,
        truncation=True,
        max_length=128,
        num_workers=4,
        top_k=None,
    )


# ---------------------------------------------------------------------------
# Step 1 — spaCy: aspect extraction (doc-based for nlp.pipe compatibility)
# ---------------------------------------------------------------------------

def normalize_aspect_tokens(chunk: spacy.tokens.Span) -> list[spacy.tokens.Token]:
    """
    Strip leading determiners/possessive pronouns from a noun chunk so that
    "the story", "this story", "a story", "story" all collapse to the same
    underlying phrase. Only strips from the front — internal determiners in
    compound phrases are left alone since they're rare and usually meaningful.
    """
    tokens = list(chunk)
    while tokens and tokens[0].pos_ in DROP_LEADING_POS:
        tokens = tokens[1:]
    return tokens


def extract_aspects_from_doc(doc: spacy.tokens.Doc) -> list[dict]:
    """
    Extract noun-chunk aspects from a pre-parsed spaCy Doc.
    Accepts a Doc (not raw text) so it is compatible with nlp.pipe().

    Returns a list of {"aspect_raw", "aspect"} dicts rather than plain strings:
      - aspect_raw: original surface text — needed later for sentence-window
                    matching in get_context_window (lemmatized text can fail
                    to match irregular surface forms, e.g. "child" vs "children").
      - aspect:     determiner-stripped, lemmatized canonical form used for
                    de-duplication and clustering.
    """
    excluded_spans = set()
    for ent in doc.ents:
        if ent.label_ in {"PERSON", "GPE", "LOC", "DATE", "TIME", "ORG", "CARDINAL"}:
            excluded_spans.update(range(ent.start, ent.end))

    seen_canonical = set()
    aspects = []
    for chunk in doc.noun_chunks:
        root = chunk.root

        if any(token.i in excluded_spans for token in chunk):
            continue

        content_tokens = normalize_aspect_tokens(chunk)
        if not content_tokens:
            continue

        raw_term = chunk.text.lower().strip()
        canonical_term = " ".join(t.lemma_.lower() for t in content_tokens).strip()

        if (
                root.dep_ in {"nsubj", "dobj", "pobj", "attr", "nsubjpass"}
                and canonical_term not in NOISE_TOKENS
                and canonical_term not in GENERIC_ASPECT_NOUNS
                and root.lemma_.lower() not in NOISE_TOKENS
                and root.lemma_.lower() not in GENERIC_ASPECT_NOUNS
                and len(canonical_term) > 2
                and not all(w in STOP_WORDS for w in canonical_term.split())
        ):
            # De-dup within this doc on the canonical form, not the raw form,
            # so "the story"/"this story" don't both survive as separate rows.
            if canonical_term in seen_canonical:
                continue
            seen_canonical.add(canonical_term)
            aspects.append({"aspect_raw": raw_term, "aspect": canonical_term})

    return aspects


# ---------------------------------------------------------------------------
# Step 3 — Context window: find the sentence containing an aspect
# ---------------------------------------------------------------------------

def get_context_window(doc: spacy.tokens.Doc, aspect: str, window: int = 1) -> str:
    """
    Return the sentence containing the aspect plus ±window surrounding sentences.
    Falls back to the full review text if aspect is not found.
    """
    sents = list(doc.sents)
    for i, sent in enumerate(sents):
        if aspect.lower() in sent.text.lower():
            start = max(0, i - window)
            end = min(len(sents), i + window + 1)
            return " ".join(s.text for s in sents[start:end])
    return doc.text


# ---------------------------------------------------------------------------
# Step 4 — Sentiment: batched inference over all (aspect, context) pairs
# ---------------------------------------------------------------------------

def build_sentiment_inputs(aspect_df: pd.DataFrame) -> list[str]:
    """
    Pre-build all model input strings in one pass using nlp.pipe for speed.
    Format: [CLS] aspect [SEP] context [SEP]

    Context lookup uses aspect_raw (the original surface form) since that's
    what actually appears in the sentence text — the lemmatized `aspect`
    column is used only as the display/dedup key and for the model input's
    [CLS] segment.
    """

    unique = aspect_df.drop_duplicates("review_idx")[["review_idx", "review"]].reset_index(drop=True)
    idx_to_aspects = aspect_df.groupby("review_idx")[["aspect", "aspect_raw"]].apply(
        lambda g: list(zip(g["aspect"], g["aspect_raw"]))
    )

    context_map: dict[tuple, str] = {}

    for doc, (_, row) in tqdm(
            zip(nlp.pipe(unique["review"].tolist(), batch_size=512), unique.iterrows()),
            total=len(unique),
            desc="Building contexts",
    ):

        for aspect, aspect_raw in idx_to_aspects.get(row["review_idx"], []):
            context_map[(row["review_idx"], aspect)] = get_context_window(doc, aspect_raw, window=1)

    return [
        f"[CLS] {row['aspect']} [SEP] {context_map[(row['review_idx'], row['aspect'])]} [SEP]"
        for _, row in aspect_df.iterrows()
    ]


def parse_sentiment_result(result: list[dict]) -> dict:
    """
    Parse a single top_k=None result into a structured dict with all polarities.
    """
    scores = {r["label"].lower(): round(r["score"], 3) for r in result}
    top_label = max(scores, key=scores.get)
    return {
        "label": top_label,
        "positive": scores.get("positive", 0.0),
        "neutral": scores.get("neutral", 0.0),
        "negative": scores.get("negative", 0.0),
        "confidence": scores.get(top_label, 0.0),
    }


def run_batched_sentiment(inputs: list[str], batch_size: int = 256) -> list[dict]:
    """
    Run sentiment inference in batches. Returns a list of parsed result dicts.
    L40S (48GB VRAM) handles batch_size=256 comfortably; increase to 512 if no OOM.
    """

    raw_results = list(tqdm(
        sentiment_model(inputs, batch_size=batch_size, top_k=None),
        total=len(inputs),
        desc="Sentiment inference"
    ))
    return [parse_sentiment_result(r) for r in raw_results]


# ---------------------------------------------------------------------------
# Step 5 — Clustering: find best K and assign cluster labels
# ---------------------------------------------------------------------------

def merge_typo_variants(aspect_df: pd.DataFrame, min_ratio: float = 90.0) -> tuple[pd.DataFrame, dict]:
    """
    Merge near-identical spellings of the same term (e.g. "delivery" vs
    "delievery") using string edit distance, NOT embeddings. Misspellings
    aren't a semantic relationship — they're the same string with a few
    characters changed — so an embedding model has no reason to score them
    highly, and calibration showed exactly that (delivery/delievery scored
    0.157-0.312 across three different embedding models, well below genuine
    synonym pairs). This pass runs first and cheaply, on canonical terms only,
    before the (more expensive) embedding-based semantic merge.

    Returns the updated aspect_df plus a {original_term: canonical_term}
    mapping so callers can see what got merged.
    """
    from rapidfuzz import fuzz

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

    # Bucket by first character before comparing — typos essentially never
    # change the first letter of a word, so this skips the vast majority of
    # pairwise comparisons for free and keeps this scaling with vocabulary
    # size rather than its square, once the unique-term count gets large
    # (e.g. running against the full corpus rather than a small sample).
    buckets: dict[str, list[str]] = {}
    for t in unique_terms:
        if t:
            buckets.setdefault(t[0], []).append(t)

    for bucket_terms in buckets.values():
        for i, a in enumerate(bucket_terms):
            for b in bucket_terms[i + 1:]:
                if abs(len(a) - len(b)) > 3:
                    continue  # cheap pre-filter, typo variants rarely differ this much in length
                if fuzz.ratio(a, b) >= min_ratio:
                    union(a, b)

    # Canonical label per merged group = the most frequent surface form,
    # so the "correct" spelling (usually more common) wins over the typo.
    term_counts = aspect_df["aspect"].value_counts().to_dict()
    groups: dict[str, list[str]] = {}
    for t in unique_terms:
        groups.setdefault(find(t), []).append(t)

    term_to_canonical = {}
    merges = {}
    for members in groups.values():
        canonical = max(members, key=lambda t: term_counts.get(t, 0))
        for t in members:
            term_to_canonical[t] = canonical
            if t != canonical:
                merges[t] = canonical

    aspect_df = aspect_df.copy()
    aspect_df["aspect"] = aspect_df["aspect"].map(term_to_canonical)

    if merges:
        print(f"  Merged {len(merges)} typo variants into their canonical spelling:")
        for original, canonical in sorted(merges.items()):
            print(f"    {original!r} -> {canonical!r}")

        # Groups with 3+ members merged via transitive chaining (A~B, B~C)
        # deserve a manual look — pairwise edit-distance similarity doesn't
        # guarantee A and C are actually similar to each other.
        chained = {c: ms for c, ms in groups.items() if len(ms) > 2}
        if chained:
            print(f"  ⚠ {len(chained)} group(s) merged 3+ terms via chaining — verify these are real typo variants:")
            for canonical_root, members in chained.items():
                print(f"    {sorted(members)}")

    return aspect_df, merges


def assign_cluster_ids(aspect_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign a cluster id to each distinct aspect string, with no semantic
    merging step. After merge_typo_variants (spelling-variant merge) and the
    lemma/determiner normalization done at extraction time, each remaining
    distinct "aspect" value is treated as its own cluster.

    Semantic synonym merging (e.g. "story" / "plot") was tried via three
    different approaches — plain radius-neighbor clustering, single-linkage
    union-find, and mutual-nearest-neighbor pairing — and consistently
    produced either zero real merges or (in the single-linkage case)
    catastrophic over-merging via chaining. Given the aspect-sentiment
    output here is a reporting/explainability layer rather than a feature
    feeding the core GHCF/GHC2F ranking model (which uses its own separate
    BERTopic pipeline, see Section 4.5), the engineering cost of a reliable
    semantic-merge step wasn't worth it relative to what it was actually
    buying: dedup + typo-correction already remove the surface-level
    fragmentation that dominated the original aspect counts.
    """
    aspect_df = aspect_df.copy()
    unique_terms = aspect_df["aspect"].unique().tolist()
    term_to_cluster = {t: cid for cid, t in enumerate(unique_terms)}
    aspect_df["cluster"] = aspect_df["aspect"].map(term_to_cluster)

    print(f"  {len(unique_terms):,} clusters from {len(unique_terms):,} unique terms "
          f"(1:1 — no semantic merging)")

    # Drop low-quality clusters (dominated by noise tokens)
    aspect_df = filter_noise_clusters(aspect_df, threshold=0.6)

    return aspect_df



def cluster_quality_score(cluster_id: int, aspect_df: pd.DataFrame) -> float:
    terms = aspect_df[aspect_df["cluster"] == cluster_id]["aspect"].tolist()
    if not terms:
        return 0.0
    clean = [
        t for t in terms
        if t not in NOISE_TOKENS and t not in GENERIC_ASPECT_NOUNS and len(t) > 3
    ]
    return len(clean) / len(terms)


def filter_noise_clusters(aspect_df: pd.DataFrame, threshold: float = 0.6) -> pd.DataFrame:
    scores = {
        cid: cluster_quality_score(cid, aspect_df)
        for cid in aspect_df["cluster"].unique()
    }
    valid = [cid for cid, score in scores.items() if score >= threshold]
    dropped = len(scores) - len(valid)
    print(f"  Dropped {dropped} noise clusters (below quality threshold {threshold})")
    return aspect_df[aspect_df["cluster"].isin(valid)].copy()


# ---------------------------------------------------------------------------
# Step 6 — Aggregate: majority-vote sentiment per (review, cluster)
# ---------------------------------------------------------------------------

def aggregate_sentiments(aspect_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge duplicate aspects within the same review+cluster via majority vote.
    Returns one row per (review_idx, cluster).
    """

    def agg_group(group):
        counts = group['sentiment'].value_counts()
        top_sentiment = counts.index[0]  # majority vote

        return pd.Series({
            'aspect': group['aspect'].value_counts().index[0],
            'sentiment': top_sentiment,
            'mentions': len(group),
            'confidence': group['confidence'].mean(),
            'prob_pos': group['prob_pos'].mean(),
            'prob_neu': group['prob_neu'].mean(),
            'prob_neg': group['prob_neg'].mean(),
        })

    return (
        aspect_df
        .groupby(['review_idx', 'cluster'])
        .apply(agg_group, include_groups=False)
        .reset_index()
    )

def load_dataset(args):
    lu, li = LabelEncoder(), LabelEncoder()
    print(sep)
    print(f"LOADING DATASET: {args.dataset_name}")
    print(sep)

    if args.file_type == "jsonl":
        df = pd.read_json(os.path.join(args.dataset_path, f"{args.dataset_name}.jsonl"), lines=True)
    else:
        df = pd.read_parquet(os.path.join(args.dataset_path, f"{args.dataset_name}.parquet"))


    df['userId'] = lu.fit_transform(df[args.user_col])
    df['itemId'] = lu.fit_transform(df[args.item_col])
    df = df.reset_index(drop=True)

    df[args.text_col] = df[args.text_col].apply(clean_text)

    if args.is_sample:
        df = df.sample(n=min(50, df.shape[0]), random_state=42).reset_index(drop=True)
        print(f"  Running on sample of {df.shape[0]} rows")
    else:
        print(f"  Total rows: {df.shape[0]:,}")


    return df

sep = "=" * 80

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--file_type", type=str, required=True)
    parser.add_argument("--text_col", type=str, required=True)
    parser.add_argument("--user_col", type=str, required=True, default='userId')
    parser.add_argument("--item_col", type=str, required=True, default='itemId')
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--is_sample", type=lambda x: x.strip().lower() == "true", default=False)
    parser.add_argument("--generate_aspects", type=lambda x: x.strip().lower() == "true", default=False)
    args = parser.parse_args()

    global nlp, sentiment_model
    nlp = spacy.load("en_core_web_sm")

    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"

    sentiment_model = load_sentiment_model_trt(model_name, batch_size=args.batch_size)

    sep = "=" * 80

    if args.generate_aspects:
        print(sep)
        print("... STARTING THE CLEANING TEXT PROCESS ...")
        print(sep)
        df  = load_dataset(args)

    else:
        df = pd.read_parquet(f'data/aspects_{args.dataset_name}.parquet')

    # ---- Step 1: Aspect extraction ------------------------------------------
    print(sep)
    print("EXTRACTING ASPECTS (spaCy nlp.pipe)")
    print(sep)
    aspect_df = process_in_chunks(df, chunk_size=200_000, n_process=40)
    # aspect_df.to_parquet(f'data/aspects_{args.dataset_name}.parquet', index=False)
    del df
    gc.collect()
    print(f"  Total aspect rows extracted: {aspect_df.shape[0]:,}")
    print(f"  Unique canonical aspects   : {aspect_df['aspect'].nunique():,}")

    # ---- Step 2: Dedup — typo-variant merge + cluster-id assignment ----------
    # No semantic (embedding-based) clustering: three separate approaches were
    # tried and either produced zero real merges or catastrophic over-merging
    # via chaining. See assign_cluster_ids() docstring for the full rationale.
    print(sep)
    print("DEDUPING ASPECTS (typo-variant merge, no semantic clustering)")
    print(sep)
    aspect_df, _typo_merges = merge_typo_variants(aspect_df, min_ratio=90.0)
    aspect_df = assign_cluster_ids(aspect_df)

    print(sep)
    print("UPDATE ASPECTS AFTER CLUSTERING")
    print(sep)
    aspect_df.to_parquet(f'data/aspects_{args.dataset_name}.parquet', index=False)


    # ---- Step 3: Sentiment -------------------------------------------------
    print(sep)
    print("SENTIMENT ANALYSIS")
    print(sep)
    inputs = build_sentiment_inputs(aspect_df)
    sentiments = run_batched_sentiment(inputs, batch_size=args.batch_size)

    aspect_df["sentiment"] = [s["label"] for s in sentiments]
    aspect_df["confidence"] = [s["confidence"] for s in sentiments]
    aspect_df["prob_pos"] = [s["positive"] for s in sentiments]
    aspect_df["prob_neu"] = [s["neutral"] for s in sentiments]
    aspect_df["prob_neg"] = [s["negative"] for s in sentiments]


    # ---- Step 4: Aggregation ------------------------------------------------
    print(sep)
    print("AGGREGATING SENTIMENTS PER CLUSTER")
    print(sep)
    aspect_df_agg = aggregate_sentiments(aspect_df)
    print(f"  Aggregated rows: {len(aspect_df_agg):,}")

    # ---- Save ---------------------------------------------------------------
    print(sep)
    print(f"SAVING RESULTS FOR: {args.dataset_name}")
    print(sep)

    agg_path = f"results/{args.dataset_name}_aspects.csv"

    aspect_df_agg.to_csv(agg_path, index=False)

    print(f"  Aggregated saved to  : {agg_path}")
    print(sep)
    print("DONE")
    print(sep)

    print("  Releasing GPU memory...")
    del sentiment_model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print(f"  GPU memory after cleanup: "
          f"{torch.cuda.memory_allocated() / 1024 ** 2:.1f}MB allocated, "
          f"{torch.cuda.memory_reserved() / 1024 ** 2:.1f}MB reserved")


if __name__ == "__main__":
    main()