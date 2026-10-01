import argparse
import json
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sep = "=" * 80
SENTIMENT_SIGN = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}


# ---------------------------------------------------------------------------
# 0. Reviews + leave-one-out split
# ---------------------------------------------------------------------------

def load_absa(path: str) -> pd.DataFrame:
    """
    Loads the precomputed aspect file (one row per (review_idx, cluster)),
    e.g. aspects/extracted_aspects/all_beauty_absa_test_aspects.csv, which
    already carries userId / itemId / polarity. Nothing is re-extracted.
    """
    absa = pd.read_csv(path)
    required = {"review_idx", "userId", "itemId", "aspect", "sentiment", "mentions", "confidence"}
    missing = required - set(absa.columns)
    if missing:
        raise KeyError(f"{path} is missing columns: {sorted(missing)}")
    absa["aspect"] = absa["aspect"].astype(str).str.lower().str.strip()
    print(f"  Loaded {len(absa):,} aspect rows from {path} "
          f"({absa.review_idx.nunique():,} reviews, {absa.userId.nunique():,} users, "
          f"{absa.itemId.nunique():,} items)")
    return absa


def build_review_table(absa: pd.DataFrame, reviews_path: Optional[str],
                       user_col: str, item_col: str,
                       raw_user_col: Optional[str] = None,
                       raw_item_col: Optional[str] = None) -> Tuple[pd.DataFrame, bool, pd.DataFrame]:
    """
    One row per review, used for |R_e| (P4), chronological order (P9) and the
    split. Two sources:

      * reviews_path given: the full review table (same row order used by the
        extraction, so review_idx = row position). It includes reviews in which
        NO aspect was found, so |R_e| counts every review of e.
        If the table carries raw_user_col / raw_item_col (the ids the
        extraction saw, e.g. a filtered + re-encoded training CSV), those are
        checked against the aspect file, and the aspect file's userId/itemId
        are replaced by the table's (model index space) via review_idx.
        Aspect rows of reviews absent from the table are dropped and counted.
      * otherwise: derived from the aspect file itself. Only reviews with at
        least one aspect exist there, so f_{e,k} in P4 becomes "share of e's
        reviews-with-aspects that mention k". Returns has_full_table=False.
    """
    if reviews_path:
        rev = pd.read_parquet(reviews_path) if reviews_path.endswith(".parquet") else pd.read_csv(reviews_path)
        rev = rev.reset_index(drop=True)
        if "review_idx" not in rev.columns:
            rev["review_idx"] = np.arange(len(rev))
        rev = rev.rename(columns={user_col: "userId", item_col: "itemId"})
        remap = bool(raw_user_col and raw_item_col and
                     raw_user_col in rev.columns and raw_item_col in rev.columns)
        ru, ri = (raw_user_col, raw_item_col) if remap else ("userId", "itemId")

        # Consistency check against the aspect file (on the ids the extraction saw).
        chk = absa[["review_idx", "userId", "itemId"]].drop_duplicates("review_idx").merge(
            rev[["review_idx", ru, ri]].rename(columns={ru: "userId_rev", ri: "itemId_rev"}),
            on="review_idx", how="inner")
        bad = ((chk.userId != chk.userId_rev) | (chk.itemId != chk.itemId_rev)).mean() if len(chk) else 1.0
        if bad > 0:
            raise ValueError(f"{100*bad:.2f}% of review_idx map to a different (user, item) in "
                             f"{reviews_path}: the files are misaligned.")

        if remap:
            n_rows, n_rev = len(absa), absa.review_idx.nunique()
            absa = absa.drop(columns=["userId", "itemId"]).merge(
                rev[["review_idx", "userId", "itemId"]], on="review_idx", how="inner")
            print(f"  Remapped aspect ids {ru}/{ri} -> userId/itemId via review_idx: kept "
                  f"{len(absa):,}/{n_rows:,} aspect rows ({absa.review_idx.nunique():,}/{n_rev:,} reviews); "
                  f"dropped {n_rows - len(absa):,} rows of reviews outside {reviews_path}")
            print(f"  Reviews in the table with >=1 extracted aspect: "
                  f"{absa.review_idx.nunique():,}/{len(rev):,} "
                  f"({100 * absa.review_idx.nunique() / max(len(rev), 1):.1f}%)")
        print(f"  Review table: {len(rev):,} reviews from {reviews_path} (aligned with aspect file)")
        return rev, True, absa

    rev = absa.drop_duplicates("review_idx")[["review_idx", "userId", "itemId"]].reset_index(drop=True)
    print(f"  Review table derived from the aspect file: {len(rev):,} reviews "
          f"(reviews with zero aspects are not visible; see P4 note)")
    return rev, False, absa


def attach_split(rev: pd.DataFrame, split_path: Optional[str], split_col: Optional[str],
                 time_col: Optional[str], allow_no_split: bool) -> pd.Series:
    """
    Priority: split file (review_idx, split) > split column in the review
    table > LOO rebuilt from time_col > everything "train" (only with
    --allow_no_split, for exploration; never for reported experiments).
    """
    if split_path:
        sp = pd.read_csv(split_path)[["review_idx", "split"]]
        s = rev[["review_idx"]].merge(sp, on="review_idx", how="left")["split"]
        n_miss = s.isna().sum()
        if n_miss:
            print(f"  ⚠ {n_miss:,} reviews not in {split_path}; treated as held-out (excluded).")
        s = s.fillna("unknown").astype(str).str.lower()
        s.index = rev.index
        print(f"  Split from {split_path}: {s.value_counts().to_dict()}")
        return s
    if (split_col and split_col in rev.columns) or (time_col and time_col in rev.columns):
        return loo_split(rev, "userId", time_col, split_col)
    if allow_no_split:
        print("  ⚠ NO SPLIT: every review is used for the profiles. Held-out reviews leak into "
              "the profiles -- use only for exploration, not for reported results.")
        return pd.Series("train", index=rev.index)
    raise ValueError("No split source. Pass --split_path, or --reviews_path with --time_col/--split_col, "
                     "or --allow_no_split for exploration only.")


def loo_split(df: pd.DataFrame, user_col: str, time_col: Optional[str],
              split_col: Optional[str]) -> pd.Series:
    """
    Returns a Series in {"train", "val", "test"} aligned with df.

    Priority:
      1. split_col already in df (reuse the exact split of the training code)
      2. LOO by time: last interaction -> test, second-to-last -> val
         (paper, Sec. 4.1). Ties broken by review_idx for determinism.
    """
    if split_col and split_col in df.columns:
        s = df[split_col].astype(str).str.lower()
        print(f"  Using existing split column {split_col!r}: {s.value_counts().to_dict()}")
        return s

    if not time_col or time_col not in df.columns:
        raise ValueError(
            "No split available: pass --split_col (existing column) or --time_col "
            "(to rebuild the LOO split). Building profiles without a split would "
            "leak held-out reviews into the profiles."
        )

    order = df.sort_values([user_col, time_col, "review_idx"], kind="mergesort")
    rank_from_end = order.groupby(user_col).cumcount(ascending=False)
    split = pd.Series("train", index=order.index)
    split[rank_from_end == 0] = "test"
    n_per_user = order.groupby(user_col)[user_col].transform("size")
    split[(rank_from_end == 1) & (n_per_user >= 3)] = "val"
    split = split.reindex(df.index)
    print(f"  LOO split by {time_col!r}: {split.value_counts().to_dict()}")
    return split


# ---------------------------------------------------------------------------
# 1. Aspect vocabulary  (Eq. P1)
# ---------------------------------------------------------------------------

@dataclass
class AspectVocabulary:
    term_to_cat: Dict[str, int]
    names: List[str]
    members: List[List[str]]

    @property
    def K(self) -> int:
        return len(self.names)


def _term_statistics(absa_train: pd.DataFrame) -> pd.DataFrame:
    """Per-term review / user / item document frequencies (train only)."""
    stats = absa_train.groupby("aspect").agg(
        n_reviews=("review_idx", "nunique"),
        n_users=("userId", "nunique"),
        n_items=("itemId", "nunique"),
        mentions=("mentions", "sum"),
    )
    return stats.sort_values("n_reviews", ascending=False)


def default_embedder(model_name: str) -> Callable[[List[str]], np.ndarray]:
    from sentence_transformers import SentenceTransformer  # lazy import
    model = SentenceTransformer(model_name)
    return lambda terms: model.encode(terms, batch_size=256, normalize_embeddings=True,
                                      show_progress_bar=True)


def build_aspect_vocabulary(absa_train: pd.DataFrame, method: str, K: int,
                            min_user_df: int, min_item_df: int, candidate_pool: int,
                            embedder: Optional[Callable] = None,
                            seed: int = 42) -> Tuple[AspectVocabulary, pd.DataFrame]:
    """
    Candidate terms must appear across >= min_user_df users AND >= min_item_df
    items. The item constraint removes item-specific terms (character names,
    titles) that describe one movie rather than a reusable aspect.

    method="topk"  : each of the K most frequent candidates is its own category.
    method="embed" : the `candidate_pool` most frequent candidates are embedded
                     and grouped with frequency-weighted KMeans into K
                     categories; each category is named after its most
                     frequent member. Terms outside the pool are discarded.
    """
    stats = _term_statistics(absa_train)
    cand = stats[(stats.n_users >= min_user_df) & (stats.n_items >= min_item_df)]
    print(f"  {len(stats):,} distinct terms -> {len(cand):,} pass "
          f"user_df>={min_user_df} & item_df>={min_item_df}")
    if len(cand) < K:
        raise ValueError(f"Only {len(cand)} candidate terms for K={K}; lower K or the df thresholds.")

    if method == "topk":
        chosen = cand.head(K).index.tolist()
        vocab = AspectVocabulary(
            term_to_cat={t: k for k, t in enumerate(chosen)},
            names=chosen, members=[[t] for t in chosen])
        return vocab, stats

    if method != "embed":
        raise ValueError(f"Unknown vocabulary method {method!r}")

    from sklearn.cluster import KMeans
    pool = cand.head(candidate_pool)
    terms = pool.index.tolist()
    emb = np.asarray(embedder(terms), dtype=np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True).clip(min=1e-12)
    km = KMeans(n_clusters=K, n_init=10, random_state=seed)
    labels = km.fit_predict(emb, sample_weight=np.log1p(pool.n_reviews.values))

    members = [[] for _ in range(K)]
    for t, k in zip(terms, labels):  # terms are already frequency-sorted
        members[k].append(t)
    # Re-index categories by total frequency so category 0 is the most common.
    freq = [pool.loc[m, "n_reviews"].sum() for m in members]
    order = np.argsort(freq)[::-1]
    members = [members[k] for k in order]
    names = [m[0] for m in members]
    term_to_cat = {t: new_k for new_k, m in enumerate(members) for t in m}
    return AspectVocabulary(term_to_cat, names, members), stats


# ---------------------------------------------------------------------------
# 2. Review-level aspect scores  (Eq. P2-P3)
# ---------------------------------------------------------------------------

def review_aspect_scores(absa: pd.DataFrame, vocab: AspectVocabulary) -> pd.DataFrame:
    """
    Collapses the ABSA rows of each review onto the K categories.

      polarity  pi_m = p+_m - p-_m                    (in [-1, 1])
      weight    w_m  = mentions_m * confidence_m
      s_{r,k} = sum_m w_m pi_m / sum_m w_m,   w_{r,k} = sum_m w_m

    Missing probabilities (classifier returned an incomplete distribution)
    fall back to sign(sentiment label) * confidence.
    """
    a = absa[absa["aspect"].isin(vocab.term_to_cat)].copy()
    a["cat"] = a["aspect"].map(vocab.term_to_cat).astype(np.int32)

    if "polarity" in a.columns:          # precomputed p+ - p- (aspect file)
        pol = a["polarity"].astype(float)
    else:
        pol = a["prob_positive"] - a["prob_negative"]
    fallback = a["sentiment"].map(SENTIMENT_SIGN).fillna(0.0) * a["confidence"]
    a["pol"] = pol.fillna(fallback).clip(-1.0, 1.0)
    a["w"] = a["mentions"].astype(float) * a["confidence"].astype(float)
    a["ws"] = a["w"] * a["pol"]

    r = a.groupby(["review_idx", "cat"], as_index=False).agg(w=("w", "sum"), ws=("ws", "sum"))
    r["s"] = (r["ws"] / r["w"].clip(lower=1e-12)).clip(-1.0, 1.0)
    return r


# ---------------------------------------------------------------------------
# 3. Entity profiles  (Eq. P4-P8)
# ---------------------------------------------------------------------------

def entity_profiles(rev_scores: pd.DataFrame, reviews_train: pd.DataFrame, entity_col: str,
                    num_entities: int, K: int, beta: float, mu: np.ndarray,
                    center: bool) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    For entity e (user or item) with training reviews R_e:

      f_{e,k}      = |{r in R_e : k in r}| / |R_e|                  (P4)
      idf_k        = log((1 + N) / (1 + df_k)) + 1                  (P5)
      alpha_hat_e  = normalize_L1(f_e * idf)                        (P5)
      W_{e,k}      = sum_{r in R_e} w_{r,k}
      s_bar_{e,k}  = (sum_r w_{r,k} s_{r,k} + beta mu_k) / (W_{e,k} + beta)   (P6)
      lambda_{e,k} = W_{e,k} / (W_{e,k} + beta)                     (P7)
      [users, optional] s_tilde_{u,k} = s_bar_{u,k} - lambda_{u,k} (b_u - mu_bar)  (P8)

    Entities with no training review get alpha=0, s_bar=mu, lambda=0, i.e.
    exactly the global prior (graceful cold start).
    """
    m = rev_scores.merge(reviews_train[["review_idx", entity_col]], on="review_idx", how="inner")
    e = m[entity_col].to_numpy()
    k = m["cat"].to_numpy()

    n_reviews = np.bincount(reviews_train[entity_col].to_numpy(), minlength=num_entities).astype(float)

    counts = np.zeros((num_entities, K))
    W = np.zeros((num_entities, K))
    WS = np.zeros((num_entities, K))
    np.add.at(counts, (e, k), 1.0)
    np.add.at(W, (e, k), m["w"].to_numpy())
    np.add.at(WS, (e, k), m["ws"].to_numpy())

    # (P4-P5) importance
    f = counts / n_reviews.clip(min=1.0)[:, None]
    active = n_reviews > 0
    df_k = (counts[active] > 0).sum(axis=0)
    idf = np.log((1.0 + active.sum()) / (1.0 + df_k)) + 1.0
    alpha = f * idf
    alpha /= alpha.sum(axis=1, keepdims=True).clip(min=1e-12)

    # (P6-P7) shrunk sentiment + reliability
    s_bar = (WS + beta * mu[None, :]) / (W + beta)
    lam = W / (W + beta)

    # (P8) leniency correction (only meaningful for users)
    if center:
        mu_bar = float((mu * np.maximum(W.sum(0), 1e-12)).sum() / np.maximum(W.sum(), 1e-12))
        b = (WS.sum(1) + beta * mu_bar) / (W.sum(1) + beta)
        s_bar = s_bar - lam * (b - mu_bar)[:, None]
    s_bar = np.clip(s_bar, -1.0, 1.0)

    profile = np.concatenate([alpha, s_bar, lam], axis=1).astype(np.float32)
    diag = {
        "entities_with_train_reviews": int(active.sum()),
        "entities_with_any_aspect": int((counts.sum(1) > 0).sum()),
        "mean_aspects_per_entity": float((counts > 0).sum(1)[active].mean()) if active.any() else 0.0,
        "mean_lambda_active": float(lam[active].mean()) if active.any() else 0.0,
    }
    return profile, diag


# ---------------------------------------------------------------------------
# 4. Review sequences for the attention profiler  (Eq. P9)
# ---------------------------------------------------------------------------

def review_sequences(rev_scores: pd.DataFrame, reviews_train: pd.DataFrame, entity_col: str,
                     num_entities: int, K: int, time_col: Optional[str],
                     max_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    x_r = [ e_{r,1} s_{r,1}, ..., e_{r,K} s_{r,K} | e_{r,1}, ..., e_{r,K} ]  in R^{2K}

    The presence half lets the attention profiler distinguish "mentioned with
    neutral sentiment" (0 | 1) from "not mentioned" (0 | 0). Only reviews with
    at least one in-vocabulary aspect are kept, ordered chronologically, and
    each entity keeps its `max_len` most recent ones.

    Stored ragged (CSR style): values[offsets[e]:offsets[e+1]] are entity e's
    review vectors. Pad per batch with AspectProfileStore.padded().
    """
    rids = np.sort(rev_scores["review_idx"].unique())
    row_of = pd.Series(np.arange(len(rids)), index=rids)
    X = np.zeros((len(rids), 2 * K), dtype=np.float32)
    rows = row_of.loc[rev_scores["review_idx"]].to_numpy()
    cats = rev_scores["cat"].to_numpy()
    X[rows, cats] = rev_scores["s"].to_numpy()
    X[rows, K + cats] = 1.0

    meta = reviews_train[reviews_train["review_idx"].isin(row_of.index)]
    sort_cols = [entity_col] + ([time_col] if time_col and time_col in meta.columns else []) + ["review_idx"]
    meta = meta.sort_values(sort_cols, kind="mergesort")
    meta = meta.groupby(entity_col, sort=False).tail(max_len)

    lengths = np.bincount(meta[entity_col].to_numpy(), minlength=num_entities)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    values = X[row_of.loc[meta["review_idx"]].to_numpy()].astype(np.float16)
    return values, offsets


# ---------------------------------------------------------------------------
# Runtime helper used by the models / dataloaders
# ---------------------------------------------------------------------------

class AspectProfileStore:
    """
    Thin accessor around the saved .npz.

      store = AspectProfileStore("profiles/amazon_aspect_profiles.npz")
      text, mask = store.padded("user", user_ids)       # (B, L, 2K), (B, L)
      ids, text, mask = store.item_corpus()             # for set_item_corpus
      P_u = store.static("user", user_ids)               # (B, 3K)
      m_ui = store.match(user_ids, item_ids)             # (B,) interpretable affinity

    Returns numpy arrays; wrap with torch.from_numpy in the collate function.
    """

    def __init__(self, path: str):
        z = np.load(path)
        self.K = int(z["K"])
        self._static = {"user": z["user_static"], "item": z["item_static"]}
        self._vals = {"user": z["user_seq_values"], "item": z["item_seq_values"]}
        self._offs = {"user": z["user_seq_offsets"], "item": z["item_seq_offsets"]}

    @property
    def seq_dim(self) -> int:
        return 2 * self.K

    @property
    def static_dim(self) -> int:
        return 3 * self.K

    def static(self, kind: str, ids) -> np.ndarray:
        return self._static[kind][np.asarray(ids)]

    def padded(self, kind: str, ids, max_len: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        ids = np.asarray(ids)
        offs, vals = self._offs[kind], self._vals[kind]
        lens = offs[ids + 1] - offs[ids]
        L = int(max(1, lens.max() if len(lens) else 1))
        if max_len:
            L = min(L, max_len)
        text = np.zeros((len(ids), L, self.seq_dim), dtype=np.float32)
        mask = np.zeros((len(ids), L), dtype=np.float32)
        for b, (e, n) in enumerate(zip(ids, lens)):
            n = min(int(n), L)
            if n:
                text[b, :n] = vals[offs[e + 1] - n: offs[e + 1]]  # most recent n
                mask[b, :n] = 1.0
        return text, mask

    def item_corpus(self, max_len: Optional[int] = None):
        offs = self._offs["item"]
        ids = np.nonzero(offs[1:] - offs[:-1])[0]
        text, mask = self.padded("item", ids, max_len)
        return ids, text, mask

    def match(self, user_ids, item_ids) -> np.ndarray:
        """m_{u,i} = sum_k alpha_hat_{u,k} * lambda_{i,k} * s_bar_{i,k}   (P10)"""
        K = self.K
        pu = self.static("user", user_ids)
        pi = self.static("item", item_ids)
        return (pu[:, :K] * pi[:, 2 * K:] * pi[:, K:2 * K]).sum(axis=1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_name", required=True)
    p.add_argument("--absa_path", default=None,
                   help="Default: aspects/extracted_aspects/{dataset_name}_absa_test_aspects.csv")
    p.add_argument("--reviews_path", default=None,
                   help="Optional full review table (same row order as extraction). Gives |R_e| "
                        "including aspect-less reviews, timestamps and/or a split column.")
    p.add_argument("--split_path", default=None,
                   help="Optional CSV with columns review_idx,split (train/val/test) -- preferred: "
                        "export it from the training pipeline so both use the same split.")
    p.add_argument("--user_col", default="userId")
    p.add_argument("--item_col", default="itemId")
    p.add_argument("--raw_user_col", default="raw_userId",
                   help="Column of --reviews_path holding the user ids the extraction saw; if present, "
                        "the aspect file is remapped to --user_col (model index space) via review_idx.")
    p.add_argument("--raw_item_col", default="raw_itemId")
    p.add_argument("--time_col", default="timestamp")
    p.add_argument("--split_col", default=None)
    p.add_argument("--allow_no_split", type=lambda x: x.strip().lower() == "true", default=False)
    p.add_argument("--num_users", type=int, default=None,
                   help="Rows of the user profile matrix; default max(userId)+1. Must match the model.")
    p.add_argument("--num_items", type=int, default=None)
    p.add_argument("--vocab_method", choices=["topk", "embed"], default="embed")
    p.add_argument("--K", type=int, default=20)
    p.add_argument("--min_user_df", type=int, default=20)
    p.add_argument("--min_item_df", type=int, default=10)
    p.add_argument("--candidate_pool", type=int, default=2000)
    p.add_argument("--embed_model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--beta", type=float, default=2.0)
    p.add_argument("--center_users", type=lambda x: x.strip().lower() == "true", default=True)
    p.add_argument("--max_seq_len", type=int, default=50)
    p.add_argument("--out_dir", default="aspects/profiles")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    print(sep); print(f"LOADING: {args.dataset_name}"); print(sep)
    absa_path = args.absa_path or os.path.join(
        "aspects", "extracted_aspects", f"{args.dataset_name}_absa_test_aspects.csv")
    absa = load_absa(absa_path)
    rev, has_full_table, absa = build_review_table(absa, args.reviews_path, args.user_col, args.item_col,
                                                   args.raw_user_col, args.raw_item_col)
    rev["split"] = attach_split(rev, args.split_path, args.split_col, args.time_col, args.allow_no_split)
    time_col = args.time_col if args.time_col in rev.columns else None
    if time_col is None:
        print("  (no timestamp: review_idx order is used as the chronological proxy in P9)")

    num_users = args.num_users or int(max(rev.userId.max(), absa.userId.max())) + 1
    num_items = args.num_items or int(max(rev.itemId.max(), absa.itemId.max())) + 1
    if rev.userId.max() >= num_users or rev.itemId.max() >= num_items:
        raise ValueError("An id exceeds --num_users/--num_items: ids in the aspect file do not "
                         "match the model's index space.")
    print(f"  Profile matrices: {num_users:,} users x {num_items:,} items "
          f"({rev.userId.nunique():,} / {rev.itemId.nunique():,} observed)")

    train = rev[rev.split == "train"]
    absa = absa.merge(rev[["review_idx", "split"]], on="review_idx", how="left")
    absa_train = absa[absa.split == "train"]
    print(f"  Aspect rows from train reviews: {len(absa_train):,}/{len(absa):,}")

    print(sep); print(f"ASPECT VOCABULARY ({args.vocab_method}, K={args.K})"); print(sep)
    embedder = default_embedder(args.embed_model) if args.vocab_method == "embed" else None
    vocab, _ = build_aspect_vocabulary(
        absa_train, args.vocab_method, args.K, args.min_user_df, args.min_item_df,
        args.candidate_pool, embedder, args.seed)
    for k, (name, mem) in enumerate(zip(vocab.names, vocab.members)):
        print(f"  [{k:02d}] {name:<20} ({len(mem)} terms) {', '.join(mem[:6])}")

    print(sep); print("REVIEW-LEVEL SCORES"); print(sep)
    rv = review_aspect_scores(absa_train, vocab)
    # Leakage guard: only train reviews may feed the profiles.
    held_out = set(rev.loc[rev.split != "train", "review_idx"])
    assert held_out.isdisjoint(rv.review_idx), "val/test reviews leaked into the aspect profiles"
    assert set(rv.review_idx) <= set(train.review_idx)
    cov = rv.review_idx.nunique() / max(len(train), 1)
    print(f"  {rv.review_idx.nunique():,}/{len(train):,} train reviews ({100*cov:.1f}%) have >=1 "
          f"in-vocabulary aspect; {len(rv)/max(rv.review_idx.nunique(),1):.2f} aspects/review"
          + ("" if has_full_table else "  [denominator = reviews with aspects only]"))

    K = vocab.K
    Wk = np.bincount(rv["cat"], weights=rv["w"], minlength=K)
    WSk = np.bincount(rv["cat"], weights=rv["w"] * rv["s"], minlength=K)
    mu = WSk / np.maximum(Wk, 1e-12)

    print(sep); print("ENTITY PROFILES"); print(sep)
    user_static, du = entity_profiles(rv, train, "userId", num_users, K, args.beta, mu, args.center_users)
    item_static, di = entity_profiles(rv, train, "itemId", num_items, K, args.beta, mu, False)
    print(f"  users: {du}")
    print(f"  items: {di}")

    user_vals, user_offs = review_sequences(rv, train, "userId", num_users, K, time_col, args.max_seq_len)
    item_vals, item_offs = review_sequences(rv, train, "itemId", num_items, K, time_col, args.max_seq_len)

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"{args.dataset_name}_aspect_profiles.npz")
    np.savez_compressed(out, K=K, global_prior=mu.astype(np.float32),
                        user_static=user_static, item_static=item_static,
                        user_seq_values=user_vals, user_seq_offsets=user_offs,
                        item_seq_values=item_vals, item_seq_offsets=item_offs)
    vocab_out = os.path.join(args.out_dir, f"{args.dataset_name}_aspect_vocab.json")
    with open(vocab_out, "w") as fh:
        json.dump({
            "config": vars(args),
            "full_review_table": has_full_table,
            "categories": [{"id": k, "name": n, "members": m, "global_polarity": float(mu[k]),
                            "train_weight": float(Wk[k])}
                           for k, (n, m) in enumerate(zip(vocab.names, vocab.members))],
            "diagnostics": {"review_coverage": cov, "users": du, "items": di},
        }, fh, indent=2)
    print(sep)
    print(f"  Saved {out}\n  Saved {vocab_out}")
    print(f"  Static profile dim = 3K = {3*K}; sequence dim (TextProfile text_dim) = 2K = {2*K}")
    print(sep)


if __name__ == "__main__":
    main()