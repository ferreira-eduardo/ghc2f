from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence


# --------------------------------------------------------------------------
# 1. Preprocessing: filter + compute the importance weight w_k
# --------------------------------------------------------------------------

def preprocess_aspects(df: pd.DataFrame, min_confidence: float = 0.5) -> pd.DataFrame:
    """Filter low-confidence extractions and compute the per-aspect weight w_k.

    Expects columns: review_idx, aspect, mentions, confidence,
                      prob_pos, prob_neu, prob_neg
    """
    df = df.loc[df["confidence"] >= min_confidence].copy()
    df = df.dropna(subset=["aspect"])

    df["polarity"] = df["prob_pos"] - df["prob_neg"]  # in [-1, 1]
    df["weight"] = (
        np.log1p(df["mentions"].astype(float))
        * df["confidence"]
        * df["polarity"].abs()
    )

    df["weight"] = df["weight"].clip(lower=1e-4)
    return df


# --------------------------------------------------------------------------
# 2. Aspect-text embedding cache
# --------------------------------------------------------------------------

def build_aspect_embedding_cache(
    df: pd.DataFrame,
    encoder,  # a sentence-transformers SentenceTransformer, or any object
              # exposing .encode(list[str]) -> np.ndarray [n, d_emb]
    aspect_col: str = "aspect",
    batch_size: int = 512,
) -> dict[str, np.ndarray]:
    """Encode each UNIQUE aspect string once and cache it.

    With 1.38M rows but only 20 clusters / a bounded vocabulary of raw
    aspect strings, deduping before encoding is a large speedup versus
    encoding every row.
    """
    unique_aspects = df[aspect_col].dropna().unique().tolist()
    embeddings = encoder.encode(
        unique_aspects,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return dict(zip(unique_aspects, embeddings))


# --------------------------------------------------------------------------
# 3. Group per-review aspect tuples into padded tensors
# --------------------------------------------------------------------------

def build_review_tensors(
    df: pd.DataFrame,
    aspect_embeddings: dict[str, np.ndarray],
    aspect_col: str = "aspect",
    max_aspects: int | None = None,
):
    """Group rows by review_idx and build a padded batch.

    Returns:
        review_idx_order: list[int]        -- row order used below
        x:  FloatTensor [N_reviews, L_max, d_emb + 3]  (padded aspect features)
        w:  FloatTensor [N_reviews, L_max]             (padded weights)
        mask: BoolTensor [N_reviews, L_max]            (True = real aspect)
    """
    d_emb = next(iter(aspect_embeddings.values())).shape[0]

    per_review_x: list[torch.Tensor] = []
    per_review_w: list[torch.Tensor] = []
    review_idx_order: list = []

    for review_idx, group in df.groupby("review_idx", sort=False):
        if max_aspects is not None:
            # keep the highest-weight aspects if a review is unusually long
            group = group.nlargest(max_aspects, "weight")

        embs = np.stack([aspect_embeddings[a] for a in group[aspect_col]])
        probs = group[["prob_pos", "prob_neu", "prob_neg"]].to_numpy(dtype=np.float32)
        feats = np.concatenate([embs, probs], axis=1)  # [L, d_emb + 3]

        per_review_x.append(torch.tensor(feats, dtype=torch.float32))
        per_review_w.append(torch.tensor(group["weight"].to_numpy(dtype=np.float32)))
        review_idx_order.append(review_idx)

    x = pad_sequence(per_review_x, batch_first=True)          # [N, L_max, d_emb+3]
    w = pad_sequence(per_review_w, batch_first=True)           # [N, L_max]
    lengths = torch.tensor([t.shape[0] for t in per_review_x])
    max_len = x.shape[1]
    mask = torch.arange(max_len)[None, :] < lengths[:, None]   # [N, L_max] bool

    return review_idx_order, x, w, mask


# --------------------------------------------------------------------------
# 4a. Baseline: confidence/mention-weighted mean pooling (no parameters)
# --------------------------------------------------------------------------

def weighted_mean_pool(x: torch.Tensor, w: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """x: [N, L, D], w: [N, L], mask: [N, L] -> [N, D]"""
    w = w.masked_fill(~mask, 0.0)
    w_sum = w.sum(dim=1, keepdim=True).clamp_min(1e-8)
    w_norm = w / w_sum                                   # [N, L]
    return torch.einsum("nl,nld->nd", w_norm, x)


# --------------------------------------------------------------------------
# 4b. Learnable attention pooling (weight prior used as an additive bias)
# --------------------------------------------------------------------------

class AttentionPool(nn.Module):
    """Permutation- and cardinality-invariant pooling over a review's
    aspect-sentiment tuples, using a learnable query and the heuristic
    weight w_k as an additive prior on the attention logits.

        alpha_k = softmax_k( q^T x_k / sqrt(D) + log(w_k) )
        review_vec = sum_k alpha_k * x_k
    """

    def __init__(self, in_dim: int, proj_dim: int | None = None):
        super().__init__()
        self.proj_dim = proj_dim or in_dim
        self.key_proj = nn.Linear(in_dim, self.proj_dim)
        self.query = nn.Parameter(torch.randn(self.proj_dim) * 0.02)
        self.out_proj = nn.Linear(in_dim, in_dim)

    def forward(self, x: torch.Tensor, w: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: [N, L, D], w: [N, L], mask: [N, L]
        keys = self.key_proj(x)                                   # [N, L, P]
        logits = torch.einsum("nlp,p->nl", keys, self.query)
        logits = logits / (self.proj_dim ** 0.5)
        logits = logits + torch.log(w.clamp_min(1e-8))             # weight as prior
        logits = logits.masked_fill(~mask, float("-inf"))

        alpha = torch.softmax(logits, dim=1)                       # [N, L]
        alpha = torch.nan_to_num(alpha, nan=0.0)                   # rows fully masked -> 0
        pooled = torch.einsum("nl,nld->nd", alpha, x)               # [N, D]
        return self.out_proj(pooled)


# --------------------------------------------------------------------------
# 5. End-to-end convenience wrapper
# --------------------------------------------------------------------------

def compute_review_representations(
    df: pd.DataFrame,
    encoder,
    method: str = "attention",   # "attention" | "weighted_mean"
    min_confidence: float = 0.5,
    max_aspects: int | None = None,
    pool_module: "AttentionPool | None" = None,
):
    """Full pipeline: raw aspect dataframe -> {review_idx: vector}.

    `pool_module` lets you pass in an already-instantiated (and, during
    training, already-optimizing) AttentionPool so it stays part of the
    GHC2F computation graph rather than being re-created each call.
    """
    df = preprocess_aspects(df, min_confidence=min_confidence)
    cache = build_aspect_embedding_cache(df, encoder)
    review_idx_order, x, w, mask = build_review_tensors(
        df, cache, max_aspects=max_aspects
    )

    if method == "weighted_mean":
        review_vecs = weighted_mean_pool(x, w, mask)
    elif method == "attention":
        if pool_module is None:
            pool_module = AttentionPool(in_dim=x.shape[-1])
        review_vecs = pool_module(x, w, mask)
    else:
        raise ValueError(f"unknown method: {method}")

    return review_idx_order, review_vecs, pool_module