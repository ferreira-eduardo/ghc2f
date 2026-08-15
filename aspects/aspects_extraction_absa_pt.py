import argparse
import gc
import os
import re
import sys

import pandas as pd
import spacy
import torch
from rapidfuzz import fuzz
from spacy.lang.en.stop_words import STOP_WORDS
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from preprocess.clean_text import clean_text
from sklearn.preprocessing import LabelEncoder

sep = "=" * 80

ASPECT_MODEL_ID = "yangheng/deberta-v3-base-end2end-absa"
SENTI_MODEL_ID = "yangheng/deberta-v3-base-absa-v1.1"

JUNK_TOKENS = STOP_WORDS

# Tracks classifier call failures so the FIRST one prints in full (for
# diagnosis) while the rest are counted silently instead of spamming stdout.
_classifier_error_shown = False
_classifier_error_count = 0


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
# Sentence segmentation (unchanged from the single-model pipeline)
# ---------------------------------------------------------------------------

def segment_sentences(reviews: list, nlp) -> list:
    result = []
    for doc in nlp.pipe(reviews, batch_size=64):
        sents = [s.text.strip() for s in doc.sents if s.text.strip()]
        result.append(sents if sents else ([doc.text] if doc.text.strip() else []))
    return result


# ---------------------------------------------------------------------------
# Stage 1: aspect span extraction (adapted from the reference script)
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


def extract_aspects_for_sentence(text: str, aspect_extractor) -> list:
    """
    Run the end2end extractor on one sentence, returning aspect spans with
    the extractor's own (often unreliable — see module docstring) polarity
    guess attached, for stage 2 to potentially override.
    """
    if not text.strip():
        return []

    try:
        ents = aspect_extractor(text)
    except Exception as e:
        print(f"  Extractor error on sentence '{text[:50]}...': {e}")
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


# ---------------------------------------------------------------------------
# Stage 2: dedicated sentiment classifier as a fallback (adapted from
# the reference script's classify_aspects)
# ---------------------------------------------------------------------------

def _estimate_prob_map(sentiment: str, confidence: float) -> dict:
    """
    When the classifier's result isn't used (never called because the
    extractor was already confident, OR called but its own score didn't
    beat the extractor's), estimate a 3-way distribution from the single
    label+confidence the way the reference script does throughout:
    the winning label gets `confidence`, the other two split the remainder
    evenly. This is an ESTIMATE, not a real softmax — `source` still tells
    you which kind of number you're looking at ("extractor" = estimated,
    "classifier_fallback" = the classifier's actual distribution).
    """
    others = [s for s in ("positive", "negative", "neutral") if s != sentiment]
    remainder = max(0.0, 1.0 - confidence) / 2
    prob_map = {s: remainder for s in others}
    prob_map[sentiment] = confidence
    return prob_map


def classify_aspects(text: str, aspects: list, sent_classifier,
                      fallback_confidence_threshold: float) -> list:
    """
    For each aspect, start from the extractor's own polarity guess. If its
    confidence is below fallback_confidence_threshold, ask the dedicated
    (text, aspect) classifier instead and use whichever result has higher
    confidence. Given real extractor confidences typically run well below
    0.8 (see module docstring), this fires for most aspects in practice —
    which is the point: the dedicated classifier is the more trustworthy
    source for polarity specifically.

    prob_positive/negative/neutral are always populated (no nulls): either
    the classifier's real distribution when it wins, or an estimate derived
    from the winning single confidence otherwise (see _estimate_prob_map).
    """
    if not aspects:
        return []

    enriched = []
    for asp in aspects:
        extractor_label = asp.get("extractor_label", "")
        extractor_confidence = asp.get("extractor_score", 0.0)

        sentiment = "neutral"
        confidence = extractor_confidence
        source = "extractor"

        parsed = normalize_sentiment_label(extractor_label)
        if parsed in ("positive", "negative", "neutral"):
            sentiment = parsed

        prob_map = None
        if extractor_confidence < fallback_confidence_threshold:
            try:
                result = sent_classifier({"text": text, "text_pair": asp["aspect"]})
                if result and isinstance(result, list) and len(result) > 0:
                    if isinstance(result[0], list):
                        result = result[0]

                    scores = {normalize_sentiment_label(d.get("label", "")): float(d.get("score", 0.0))
                              for d in result if isinstance(d, dict)}
                    if scores:
                        best_label = max(scores, key=scores.get)
                        best_score = scores[best_label]

                        if best_score > confidence:
                            sentiment = best_label
                            confidence = best_score
                            source = "classifier_fallback"
                            prob_map = {
                                "positive": scores.get("positive"),
                                "negative": scores.get("negative"),
                                "neutral": scores.get("neutral"),
                            }
            except Exception as e:
                global _classifier_error_shown, _classifier_error_count
                _classifier_error_count += 1
                if not _classifier_error_shown:
                    print(f"  ⚠ Classifier call failed (showing first occurrence only; "
                          f"further failures counted silently): {type(e).__name__}: {e}")
                    _classifier_error_shown = True

        # No nulls: if the classifier's result wasn't used (never attempted,
        # attempted but lost, or errored), estimate a distribution from the
        # winning single confidence instead of leaving these blank.
        if prob_map is None:
            prob_map = _estimate_prob_map(sentiment, confidence)

        enriched.append({
            **asp,
            "sentiment": sentiment,
            "confidence": confidence,
            "source": source,
            "prob_positive": prob_map["positive"],
            "prob_negative": prob_map["negative"],
            "prob_neutral": prob_map["neutral"],
        })

    return enriched


# ---------------------------------------------------------------------------
# Full extraction pipeline: sentence-split -> extract -> classify -> flatten
# ---------------------------------------------------------------------------

def extract_absa_records(reviews: list, aspect_extractor, sent_classifier, nlp,
                          fallback_confidence_threshold: float = 0.8) -> pd.DataFrame:
    records = []
    print("  Segmenting reviews into sentences...")
    sentences_per_review = segment_sentences(reviews, nlp)
    total_sentences = sum(len(s) for s in sentences_per_review)
    print(f"  {len(reviews):,} reviews -> {total_sentences:,} sentences")

    fallback_count = 0
    total_count = 0

    for idx, sentences in enumerate(tqdm(sentences_per_review, desc="ABSA extraction")):
        for sentence in sentences:
            aspects = extract_aspects_for_sentence(sentence, aspect_extractor)
            aspects = classify_aspects(sentence, aspects, sent_classifier, fallback_confidence_threshold)

            for a in aspects:
                total_count += 1
                if a["source"] == "classifier_fallback":
                    fallback_count += 1
                records.append({
                    "review_idx": idx,
                    "aspect": a["aspect"].lower(),
                    "sentiment": a["sentiment"],
                    "score": a["confidence"],
                    "source": a["source"],
                    "prob_positive": a["prob_positive"],
                    "prob_negative": a["prob_negative"],
                    "prob_neutral": a["prob_neutral"],
                })

    if total_count:
        print(f"  Sentiment source: {fallback_count:,}/{total_count:,} "
              f"({100 * fallback_count / total_count:.1f}%) used the dedicated "
              f"classifier fallback rather than the extractor's own polarity guess")

    global _classifier_error_count
    if _classifier_error_count:
        print(f"  ⚠ Classifier call failed {_classifier_error_count:,} time(s) during this run "
              f"(first occurrence printed above) — those aspects kept the extractor's own guess instead.")

    return pd.DataFrame(records, columns=["review_idx", "aspect", "sentiment", "score",
                                           "source", "prob_positive", "prob_negative", "prob_neutral"])


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
    lu = LabelEncoder()
    print(sep)
    print(f"LOADING DATASET: {args.dataset_name}")
    print(sep)

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--file_type", type=str, required=True)
    parser.add_argument("--user_col", type=str, required=True, default="userId")
    parser.add_argument("--text_col", type=str, required=True, default="review")
    parser.add_argument("--item_col", type=str, required=True, default="itemId")
    parser.add_argument("--is_sample", type=lambda x: x.strip().lower() == "true", default=False)
    parser.add_argument("--exclude_named_entities", type=lambda x: x.strip().lower() == "true", default=False)
    parser.add_argument("--fallback_confidence_threshold", type=float, default=0.8,
                         help="Below this extractor confidence, defer to the dedicated "
                              "sentiment classifier instead (default matches the reference script).")
    args = parser.parse_args()

    print(sep)
    print("LOADING ABSA MODELS (extractor + dedicated sentiment classifier)")
    print(sep)
    device = 0 if torch.cuda.is_available() else -1

    tok_asp = AutoTokenizer.from_pretrained(SENTI_MODEL_ID, use_fast=True)
    mdl_asp = AutoModelForTokenClassification.from_pretrained(ASPECT_MODEL_ID)
    aspect_extractor = pipeline(
        task="token-classification",
        model=mdl_asp,
        tokenizer=tok_asp,
        aggregation_strategy="simple",
        device=device,
    )

    sent_classifier = pipeline(
        task="text-classification",
        model=SENTI_MODEL_ID,
        device=device,
        top_k=None,
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
                      f"The fallback may not trigger correctly — check classify_aspects' label parsing.")
    except Exception as e:
        print(f"  ⚠ SELF-TEST FAILED: {type(e).__name__}: {e}")
        print("  The classifier fallback will not work for this run — every aspect will fall "
              "back to the extractor's own (less reliable) polarity guess. Fix this before "
              "trusting the output.")

    nlp = spacy.load("en_core_web_sm")

    df = load_dataset(args)

    print(sep)
    print("EXTRACTING ASPECTS (two-stage: extractor + classifier fallback, sentence-by-sentence)")
    print(sep)
    aspect_df = extract_absa_records(
        df[args.text_col].tolist(), aspect_extractor, sent_classifier, nlp,
        fallback_confidence_threshold=args.fallback_confidence_threshold,
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