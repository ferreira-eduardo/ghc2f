import argparse
import gc, os, re, sys
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
    a streaming iterable with num_workers>0. Plain list input to a pipeline
    is processed with num_workers=0 (synchronous): CPU tokenization for the
    next batch blocks until the GPU finishes the current one. Wrapping in a
    Dataset lets the pipeline's internal DataLoader prefetch/tokenize the
    next batch on a worker process while the GPU is still busy.
    """
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

# Both models are DeBERTa-v3-base; 512 is the standard max sequence length
# for this family. Some checkpoints don't ship a usable model_max_length in
# their tokenizer config, which is why the pipeline warns "asking to
# truncate but no maximum length is provided" and then silently skips
# truncation. Setting this explicitly makes truncation actually happen
# (protects against rare very-long sentences erroring out) instead of being
# a no-op.
DEFAULT_MAX_LENGTH = 512

JUNK_TOKENS = STOP_WORDS

_classifier_error_shown = False
_classifier_error_count = 0
_extractor_error_shown = False
_extractor_error_count = 0


# ---------------------------------------------------------------------------
# Aspect-level noise filtering (unchanged from the single-model pipeline)
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
    """
    Strip tagging-scheme prefixes ("asp-", "aspect-", "b-", "i-") so both
    the extractor's own labels (e.g. "ASP-Neutral") and the classifier's
    labels (e.g. "Positive") normalize to plain lowercase values.
    """
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
    """
    Sentence-splits with nlp.pipe. Two speed knobs beyond a single-process,
    full-pipeline pass:

    - n_process > 1 parallelizes across CPU processes. The earlier
      single-model script (aspects_extraction.py) used n_process=40 for
      exactly this reason — this stage is CPU-bound and was otherwise
      single-threaded here.
    - Only the components sentence boundaries actually depend on (tok2vec +
      parser, for en_core_web_sm) stay active; tagger/ner/lemmatizer/
      attribute_ruler are disabled for THIS call only. `disable` here does
      not touch the shared `nlp` object's pipeline permanently, so later
      calls elsewhere (e.g. is_named_entity(), which needs NER) still get
      the full pipeline.
    """
    keep = {"tok2vec", "parser", "senter", "sentencizer"}
    disable = [name for name in nlp.pipe_names if name not in keep]

    result = []
    for doc in nlp.pipe(reviews, batch_size=batch_size, n_process=n_process, disable=disable):
        sents = [s.text.strip() for s in doc.sents if s.text.strip()]
        result.append(sents if sents else ([doc.text] if doc.text.strip() else []))
    return result


# ---------------------------------------------------------------------------
# Stage 1: aspect span extraction (post-processing only — the model call
# itself now happens in run_aspect_extractor_batched, once per chunk of
# many sentences instead of once per sentence)
# ---------------------------------------------------------------------------

def _clean_aspect(s: str) -> str:
    """Strip leading/trailing punctuation and whitespace from an aspect span."""
    if not s:
        return ""
    return re.sub(r'^[\s.,;:!?()\[\]{}"\']+|[\s.,;:!?()\[\]{}"\']+$', "", s).strip()


def _locate_span(text: str, phrase: str):
    """
    Locate a phrase's character offsets in text, case-sensitive then
    case-insensitive. Needed because aggregation_strategy="simple" doesn't
    always return start/end for every token-classification pipeline output.
    """
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
    """
    Turn one sentence's raw extractor output (already computed by the batched
    pipeline call) into cleaned aspect-span dicts. Pure post-processing, no
    model call here — that's what makes it cheap to run per-sentence even
    though the model call itself is now batched.
    """
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
        })

    return aspects


def run_aspect_extractor_batched(sentences: list, aspect_extractor, batch_size: int,
                                  chunk_size: int, num_workers: int = 0) -> list:
    """
    Runs the extractor over `sentences` as GPU batches. Two optimizations
    beyond plain chunked batching:

    1. Length-sorted batching: sentences are processed in ascending length
       order, so each batch pads only to the longest sentence WITHIN that
       batch rather than to whatever the longest sentence in an arbitrarily
       shuffled chunk happens to be. Reviews mix very short and very long
       sentences, so this meaningfully cuts wasted padded compute. Results
       are written back to each sentence's original position, so the
       returned list is aligned 1:1 with the input order regardless.
    2. num_workers > 0 lets tokenization for the next batch run on a CPU
       worker while the GPU is still processing the current one (overlap),
       instead of blocking on tokenization every batch. Default 0 keeps the
       original synchronous, safest behavior.

    `chunk_size` still bounds how much is held in memory / re-created as a
    DataLoader at once; `batch_size` is the actual GPU mini-batch size.
    """
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
# Stage 2: sentiment — extractor's own guess, then batched classifier
# fallback for anything below the confidence threshold
# ---------------------------------------------------------------------------

def _estimate_prob_map(sentiment: str, confidence: float) -> dict:
    """
    When the classifier's result isn't used (never called because the
    extractor was already confident, OR called but its own score didn't
    beat the extractor's, OR the call failed), estimate a 3-way distribution
    from the single label+confidence: the winning label gets `confidence`,
    the other two split the remainder evenly. This is an ESTIMATE, not a
    real softmax — `source` still tells you which kind of number you're
    looking at ("extractor" = estimated, "classifier_fallback" = the
    classifier's actual distribution).
    """
    others = [s for s in ("positive", "negative", "neutral") if s != sentiment]
    remainder = max(0.0, 1.0 - confidence) / 2
    prob_map = {s: remainder for s in others}
    prob_map[sentiment] = confidence
    return prob_map


def attach_extractor_polarity(aspects: list) -> list:
    """
    Sets each aspect's baseline sentiment/confidence/source/prob fields from
    the extractor's own polarity guess. Pure Python — no model call. This is
    the "before fallback" state; apply_classifier_result() may later
    overwrite these fields in place if the batched fallback call wins.
    """
    enriched = []
    for asp in aspects:
        extractor_label = asp.get("extractor_label", "")
        extractor_confidence = asp.get("extractor_score", 0.0)

        sentiment = "neutral"
        parsed = normalize_sentiment_label(extractor_label)
        if parsed in ("positive", "negative", "neutral"):
            sentiment = parsed

        confidence = extractor_confidence
        prob_map = _estimate_prob_map(sentiment, confidence)

        enriched.append({
            **asp,
            "sentiment": sentiment,
            "confidence": confidence,
            "source": "extractor",
            "prob_positive": prob_map["positive"],
            "prob_negative": prob_map["negative"],
            "prob_neutral": prob_map["neutral"],
        })
    return enriched


def apply_classifier_result(aspect_dict: dict, raw_result) -> None:
    """
    Mutates aspect_dict in place with the classifier's result for one
    (text, aspect) pair, IF it beats the extractor's own confidence and the
    result is well-formed. raw_result is None on a failed call (see
    run_classifier_fallback_batched), and is left as a no-op here — the
    aspect keeps whatever attach_extractor_polarity already set.
    """
    if not raw_result:
        return
    result = raw_result
    if isinstance(result, list) and len(result) > 0 and isinstance(result[0], list):
        result = result[0]
    if not isinstance(result, list):
        return

    scores = {normalize_sentiment_label(d.get("label", "")): float(d.get("score", 0.0))
              for d in result if isinstance(d, dict)}
    if not scores:
        return

    best_label = max(scores, key=scores.get)
    best_score = scores[best_label]

    if best_score > aspect_dict["confidence"]:
        aspect_dict["sentiment"] = best_label
        aspect_dict["confidence"] = best_score
        aspect_dict["source"] = "classifier_fallback"
        aspect_dict["prob_positive"] = scores.get("positive")
        aspect_dict["prob_negative"] = scores.get("negative")
        aspect_dict["prob_neutral"] = scores.get("neutral")


def run_classifier_fallback_batched(all_aspects_per_sentence: list, flat_sentences: list,
                                     sent_classifier, fallback_confidence_threshold: float,
                                     batch_size: int, chunk_size: int, num_workers: int = 0) -> None:
    """
    Collects every (sentence, aspect) pair across the WHOLE corpus whose
    extractor confidence is below threshold, then classifies them together
    in large batches — instead of one sent_classifier(...) call per aspect
    as they're encountered sentence-by-sentence. Mutates
    all_aspects_per_sentence in place via apply_classifier_result.

    Same two optimizations as run_aspect_extractor_batched: pairs are
    processed in ascending (text + aspect) length order to reduce padding
    waste (order doesn't need restoring here — each pair already carries
    its own (sentence_i, aspect_j) write-back location), and num_workers>0
    overlaps CPU tokenization with GPU inference.
    """
    global _classifier_error_shown, _classifier_error_count

    fallback_pairs = []      # [{"text": ..., "text_pair": ...}, ...]
    fallback_locations = []  # [(sentence_i, aspect_j), ...] to write results back to

    for si, aspects in enumerate(all_aspects_per_sentence):
        for aj, asp in enumerate(aspects):
            if asp["confidence"] < fallback_confidence_threshold:
                fallback_pairs.append({"text": flat_sentences[si], "text_pair": asp["aspect"]})
                fallback_locations.append((si, aj))

    total_aspects = sum(len(a) for a in all_aspects_per_sentence)
    print(f"  {len(fallback_pairs):,}/{total_aspects:,} aspect(s) need classifier fallback "
          f"(extractor confidence < {fallback_confidence_threshold})")

    if not fallback_pairs:
        return

    order = sorted(range(len(fallback_pairs)),
                    key=lambda k: len(fallback_pairs[k]["text"]) + len(fallback_pairs[k]["text_pair"]))

    pbar = tqdm(total=len(order), desc="Sentiment classification (batched, length-sorted)")
    for i in range(0, len(order), chunk_size):
        idx_chunk = order[i:i + chunk_size]
        pair_chunk = [fallback_pairs[k] for k in idx_chunk]
        try:
            batch_out = list(sent_classifier(_ListDataset(pair_chunk), batch_size=batch_size, num_workers=num_workers))
        except Exception as e:
            _classifier_error_count += len(pair_chunk)
            if not _classifier_error_shown:
                print(f"  ⚠ Classifier batch call failed (showing first occurrence only; "
                      f"further failures counted silently): {type(e).__name__}: {e}")
                _classifier_error_shown = True
            batch_out = [None] * len(pair_chunk)

        for k, result in zip(idx_chunk, batch_out):
            si, aj = fallback_locations[k]
            apply_classifier_result(all_aspects_per_sentence[si][aj], result)
        pbar.update(len(idx_chunk))
    pbar.close()


# ---------------------------------------------------------------------------
# Full extraction pipeline: sentence-split -> batched extract -> batched
# classify fallback -> flatten
# ---------------------------------------------------------------------------

def extract_absa_records(reviews: list, aspect_extractor, sent_classifier, nlp,
                          fallback_confidence_threshold: float = 0.8,
                          extract_batch_size: int = 64, classify_batch_size: int = 64,
                          chunk_size: int = 4000, spacy_n_process: int = 1,
                          pipeline_num_workers: int = 0) -> pd.DataFrame:
    print("  Segmenting reviews into sentences...")
    sentences_per_review = segment_sentences(reviews, nlp, n_process=spacy_n_process)
    total_sentences = sum(len(s) for s in sentences_per_review)
    print(f"  {len(reviews):,} reviews -> {total_sentences:,} sentences")

    # Flatten review->sentences into one list with review_idx tracked in
    # parallel. This is what makes batching possible: both model calls below
    # operate on this single flat list instead of being nested inside a
    # per-review Python loop.
    flat_sentences, flat_review_idx = [], []
    for ridx, sents in enumerate(sentences_per_review):
        for s in sents:
            flat_sentences.append(s)
            flat_review_idx.append(ridx)

    # ---- Stage 1: batched, length-sorted aspect-span extraction ----
    ents_per_sentence = run_aspect_extractor_batched(
        flat_sentences, aspect_extractor, batch_size=extract_batch_size,
        chunk_size=chunk_size, num_workers=pipeline_num_workers,
    )

    all_aspects_per_sentence = [
        attach_extractor_polarity(extract_aspects_from_entities(text, ents))
        for text, ents in zip(flat_sentences, ents_per_sentence)
    ]

    # ---- Stage 2: batched, length-sorted sentiment classifier fallback ----
    run_classifier_fallback_batched(
        all_aspects_per_sentence, flat_sentences, sent_classifier,
        fallback_confidence_threshold, batch_size=classify_batch_size,
        chunk_size=chunk_size, num_workers=pipeline_num_workers,
    )

    # ---- Flatten to records, columnar (dict-of-lists) rather than growing
    # a list of 1M+ per-row dicts — pd.DataFrame(dict_of_lists) avoids the
    # per-row dict overhead pd.DataFrame(list_of_dicts) pays at this scale.
    col_review_idx, col_aspect, col_sentiment, col_score, col_source = [], [], [], [], []
    col_prob_pos, col_prob_neg, col_prob_neu = [], [], []
    fallback_count = 0
    total_count = 0
    for si, aspects in enumerate(all_aspects_per_sentence):
        ridx = flat_review_idx[si]
        for a in aspects:
            total_count += 1
            if a["source"] == "classifier_fallback":
                fallback_count += 1
            col_review_idx.append(ridx)
            col_aspect.append(a["aspect"].lower())
            col_sentiment.append(a["sentiment"])
            col_score.append(a["confidence"])
            col_source.append(a["source"])
            col_prob_pos.append(a["prob_positive"])
            col_prob_neg.append(a["prob_negative"])
            col_prob_neu.append(a["prob_neutral"])

    if total_count:
        print(f"  Sentiment source: {fallback_count:,}/{total_count:,} "
              f"({100 * fallback_count / total_count:.1f}%) used the dedicated "
              f"classifier fallback rather than the extractor's own polarity guess")

    if _extractor_error_count:
        print(f"  ⚠ Extractor call failed for {_extractor_error_count:,} sentence(s) during this run "
              f"(first occurrence printed above) — those sentences yielded no aspects.")
    if _classifier_error_count:
        print(f"  ⚠ Classifier call failed for {_classifier_error_count:,} aspect(s) during this run "
              f"(first occurrence printed above) — those aspects kept the extractor's own guess instead.")

    return pd.DataFrame({
        "review_idx": col_review_idx,
        "aspect": col_aspect,
        "sentiment": col_sentiment,
        "score": col_score,
        "source": col_source,
        "prob_positive": col_prob_pos,
        "prob_negative": col_prob_neg,
        "prob_neutral": col_prob_neu,
    })


# ---------------------------------------------------------------------------
# Typo-variant merge (unchanged)
# ---------------------------------------------------------------------------

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

    return aspect_df, merges


def cluster_quality_score(cluster_id, aspect_df: pd.DataFrame) -> float:
    terms = aspect_df[aspect_df["cluster"] == cluster_id]["aspect"].tolist()
    if not terms:
        return 0.0
    clean = [t for t in terms if t not in JUNK_TOKENS and len(t) > 2]
    return len(clean) / len(terms)


def filter_noise_clusters(aspect_df: pd.DataFrame, threshold: float = 0.6) -> pd.DataFrame:
    scores = {cid: cluster_quality_score(cid, aspect_df) for cid in aspect_df["cluster"].unique()}
    valid = [cid for cid, s in scores.items() if s >= threshold]
    dropped = len(scores) - len(valid)
    if dropped:
        print(f"  Dropped {dropped} noise cluster(s) (below quality threshold {threshold})")
    return aspect_df[aspect_df["cluster"].isin(valid)].copy()


def assign_cluster_ids(aspect_df: pd.DataFrame) -> pd.DataFrame:
    aspect_df = aspect_df.copy()
    unique_terms = aspect_df["aspect"].unique().tolist()
    term_to_cluster = {t: cid for cid, t in enumerate(unique_terms)}
    aspect_df["cluster"] = aspect_df["aspect"].map(term_to_cluster)
    print(f"  {len(unique_terms):,} clusters from {len(unique_terms):,} unique terms (1:1)")
    return filter_noise_clusters(aspect_df, threshold=0.6)


def aggregate_absa_results(aspect_df: pd.DataFrame) -> pd.DataFrame:
    """
    Majority-vote sentiment + mean confidence per (review_idx, cluster).
    prob_positive/negative/neutral are averaged with NaNs skipped — rows
    where the extractor's own guess was trusted (no classifier fallback)
    contribute no probability data, which is reflected honestly rather
    than backfilled with an assumed distribution.
    """
    def agg_group(group):
        top_sentiment = group["sentiment"].value_counts().index[0]
        return pd.Series({
            "aspect": group["aspect"].value_counts().index[0],
            "sentiment": top_sentiment,
            "mentions": len(group),
            "confidence": group["score"].mean(),
            "pct_classifier_fallback": (group["source"] == "classifier_fallback").mean(),
            "prob_positive": group["prob_positive"].mean(),
            "prob_negative": group["prob_negative"].mean(),
            "prob_neutral": group["prob_neutral"].mean(),
        })

    return (
        aspect_df
        .groupby(["review_idx", "cluster"])
        .apply(agg_group, include_groups=False)
        .reset_index()
    )


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
        return df

    lu = LabelEncoder()

    if args.file_type == "jsonl":
        df = pd.read_json(os.path.join(args.dataset_path, f"{args.dataset_name}.jsonl"), lines=True)
    else:
        df = pd.read_parquet(os.path.join(args.dataset_path, f"{args.dataset_name}.parquet"))

    df["userId"] = lu.fit_transform(df[args.user_col])
    df["itemId"] = lu.fit_transform(df[args.item_col])
    df = df.reset_index(drop=True)
    df[args.text_col] = df[args.text_col].apply(clean_text)

    if args.is_sample:
        df = df.sample(n=min(50, df.shape[0]), random_state=42).reset_index(drop=True)
        print(f"  Running on sample of {df.shape[0]} rows")
    else:
        print(f"  Total rows: {df.shape[0]:,}")

    return df


def _ensure_max_length(tokenizer, default=DEFAULT_MAX_LENGTH):
    """
    Some checkpoints' tokenizer configs leave model_max_length at a sentinel
    (e.g. int(1e30)) instead of the model's real limit, which is what
    triggers "Asking to truncate to max_length but no maximum length is
    provided" — truncation is silently skipped in that case, not just
    warned about. Clamping it here makes truncation actually happen.
    """
    if tokenizer.model_max_length is None or tokenizer.model_max_length > 100_000:
        tokenizer.model_max_length = default
    return tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--file_type", type=str, default=None)
    parser.add_argument("--user_col", type=str, default=None)
    parser.add_argument("--text_col", type=str, default=None)
    parser.add_argument("--item_col", type=str, default=None)
    parser.add_argument("--from_prepared_csv", type=lambda x: x.strip().lower() == "true", default=False,
                         help="Load datasets/{dataset_name}.csv (from build_review_dataset.py) "
                              "instead of re-deriving columns from the raw file. Skips "
                              "--dataset_path/--file_type/--user_col/--item_col/--text_col.")
    parser.add_argument("--is_sample", type=lambda x: x.strip().lower() == "true", default=False)
    parser.add_argument("--exclude_named_entities", type=lambda x: x.strip().lower() == "true", default=False)
    parser.add_argument("--fallback_confidence_threshold", type=float, default=0.8,
                         help="Below this extractor confidence, defer to the dedicated "
                              "sentiment classifier instead (default matches the reference script).")
    parser.add_argument("--extract_batch_size", type=int, default=512,
                         help="GPU mini-batch size for the aspect extractor pipeline call.")
    parser.add_argument("--classify_batch_size", type=int, default=512,
                         help="GPU mini-batch size for the sentiment classifier pipeline call.")
    parser.add_argument("--chunk_size", type=int, default=4000,
                         help="How many sentences/pairs are held in Python memory and passed "
                              "to the pipeline per outer chunk (the pipeline sub-batches this "
                              "internally at --*_batch_size). Lower this if you hit host-RAM "
                              "pressure; raise --*_batch_size if you have GPU headroom instead.")
    parser.add_argument("--fp16", type=lambda x: x.strip().lower() == "true", default=True,
                         help="Load both models in fp16 on GPU for faster inference (no effect on CPU).")
    parser.add_argument("--spacy_n_process", type=int, default=40,
                         help="CPU processes for spaCy sentence segmentation (nlp.pipe n_process). "
                              "The earlier single-model script used 40 on a multi-core machine — raise "
                              "this from the default of 1 if segmentation is a bottleneck.")
    parser.add_argument("--pipeline_num_workers", type=int, default=0,
                         help="DataLoader workers for both HF pipelines, letting CPU tokenization for "
                              "the next batch overlap with GPU inference on the current one. 0 = "
                              "synchronous (safe default); try 2-4 if the GPU is idling between batches.")
    args = parser.parse_args()

    if not args.from_prepared_csv:
        missing = [n for n in ("dataset_path", "file_type", "user_col", "item_col", "text_col")
                   if getattr(args, n) is None]
        if missing:
            parser.error(f"--{missing[0].replace('_', '-')} is required unless --from_prepared_csv=True "
                         f"(missing: {missing})")

    print(sep)
    print("LOADING ABSA MODELS (extractor + dedicated sentiment classifier)")
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

    print(sep)
    print("SELF-TEST: verifying classifier pair-input call works before running the full dataset")
    print(sep)
    try:
        test_result = sent_classifier({"text": "The food was exceptional, although the service was slow.",
                                        "text_pair": "food"})
        print(f"  Raw classifier output: {test_result}")
        if isinstance(test_result, list) and len(test_result) > 0:
            flat = test_result[0] if isinstance(test_result[0], list) else test_result
            labels = {normalize_sentiment_label(d.get("label", "")) for d in flat if isinstance(d, dict)}
            if {"positive", "negative", "neutral"} & labels:
                print("  ✓ Self-test passed — classifier fallback should work during the full run.")
            else:
                print(f"  ⚠ Self-test ran but returned unexpected labels: {labels}. "
                      f"The fallback may not trigger correctly — check apply_classifier_result's label parsing.")
    except Exception as e:
        print(f"  ⚠ SELF-TEST FAILED: {type(e).__name__}: {e}")
        print("  The classifier fallback will not work for this run — every aspect will fall "
              "back to the extractor's own (less reliable) polarity guess. Fix this before "
              "trusting the output.")

    print(sep)
    print("SELF-TEST: verifying batched (list-input) calls work for both pipelines")
    print(sep)
    try:
        batch_probe = aspect_extractor(
            ["The food was exceptional.", "Service was slow but the ambience was great."],
            batch_size=2,
        )
        assert isinstance(batch_probe, list) and len(batch_probe) == 2
        print("  ✓ Aspect extractor accepts batched list input.")
    except Exception as e:
        print(f"  ⚠ BATCHED EXTRACTOR SELF-TEST FAILED: {type(e).__name__}: {e}")
        print("  Falling back to batch_size=1 would restore the original ~unbatched behavior — "
              "check your transformers version if this fails.")

    try:
        batch_probe = sent_classifier(
            [{"text": "The food was exceptional.", "text_pair": "food"},
             {"text": "Service was slow.", "text_pair": "service"}],
            batch_size=2,
        )
        assert isinstance(batch_probe, list) and len(batch_probe) == 2
        print("  ✓ Sentiment classifier accepts batched list-of-pairs input.")
    except Exception as e:
        print(f"  ⚠ BATCHED CLASSIFIER SELF-TEST FAILED: {type(e).__name__}: {e}")
        print("  If this fails, the fallback stage will error per-chunk and every aspect in "
              "that chunk will keep the extractor's own guess — check your transformers version.")

    nlp = spacy.load("en_core_web_sm")

    df = load_dataset(args)

    print(sep)
    print("EXTRACTING ASPECTS (two-stage, batched: extractor + classifier fallback)")
    print(sep)
    aspect_df = extract_absa_records(
        df[args.text_col].tolist(), aspect_extractor, sent_classifier, nlp,
        fallback_confidence_threshold=args.fallback_confidence_threshold,
        extract_batch_size=args.extract_batch_size,
        classify_batch_size=args.classify_batch_size,
        chunk_size=args.chunk_size,
        spacy_n_process=args.spacy_n_process,
        pipeline_num_workers=args.pipeline_num_workers,
    )
    print(f"  Total aspect mentions extracted: {len(aspect_df):,}")

    print(sep)
    print("FILTERING JUNK ASPECTS")
    print(sep)
    before = len(aspect_df)
    aspect_df = aspect_df[~aspect_df["aspect"].apply(is_junk_aspect)].reset_index(drop=True)
    print(f"  Dropped {before - len(aspect_df):,} junk aspect(s)")

    if args.exclude_named_entities:
        before = len(aspect_df)
        aspect_df = aspect_df[~aspect_df["aspect"].apply(lambda t: is_named_entity(t, nlp))].reset_index(drop=True)
        print(f"  Dropped {before - len(aspect_df):,} named-entity aspect(s)")

    print(sep)
    print("DEDUPING ASPECTS (typo-variant merge)")
    print(sep)
    aspect_df, _typo_merges = merge_typo_variants(aspect_df, min_ratio=90.0)
    aspect_df = assign_cluster_ids(aspect_df)

    print(sep)
    print("AGGREGATING PER CLUSTER")
    print(sep)
    aspect_df_agg = aggregate_absa_results(aspect_df)
    print(f"  Aggregated rows: {len(aspect_df_agg):,}")

    print(sep)
    print(f"SAVING RESULTS FOR: {args.dataset_name}")
    print(sep)
    os.makedirs("results", exist_ok=True)
    agg_path = f"results/{args.dataset_name}_absa_test_aspects.csv"
    aspect_df_agg.to_csv(agg_path, index=False)
    print(f"  Aggregated saved to  : {agg_path}")

    print(sep)
    print("DONE")
    print(sep)

    print("  Releasing GPU memory...")
    del aspect_extractor, sent_classifier
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print(f"  GPU memory after cleanup: "
              f"{torch.cuda.memory_allocated() / 1024 ** 2:.1f}MB allocated, "
              f"{torch.cuda.memory_reserved() / 1024 ** 2:.1f}MB reserved")


if __name__ == "__main__":
    main()