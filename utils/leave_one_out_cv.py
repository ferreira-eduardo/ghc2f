import argparse
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

TIME_COL_CANDIDATES = ["timestamp", "unixReviewTime", "unix_time", "time", "date", "review_date"]
USER_COL_CANDIDATES = ["userId", "user_id", "user"]
ITEM_COL_CANDIDATES = ["itemId", "item_id", "item", "movieId"]


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------

def loo_labels(df: pd.DataFrame, k: int, n_folds: int = 5) -> np.ndarray:
    """One label per row of df (aligned with df.index): train / val / test / unused."""
    d = df.sort_values(["userId", "timestamp", "review_idx"], kind="mergesort")
    pos = d.groupby("userId").cumcount(ascending=False)            # 0 = most recent
    n = d.groupby("userId")["userId"].transform("size")
    ok = n >= n_folds + 2                                          # same users in every fold
    lab = np.select([~ok, pos < k, pos == k, pos == k + 1],
                    ["unused", "unused", "test", "val"], default="train")
    return pd.Series(lab, index=d.index).reindex(df.index).values


def get_loocv_fold_normalized(df: pd.DataFrame, k: int, n_folds: int = 5):
    """Train/val/test with z-scored relevance; stats from TRAIN only (for the recommender)."""
    split = loo_labels(df, k, n_folds)
    train, val, test = (df[split == s].copy() for s in ("train", "val", "test"))
    st = train.groupby("userId")["rating"].agg(train_mean="mean", train_std="std").reset_index()
    st["train_std"] = st["train_std"].replace(0, 1.0).fillna(1.0)

    def norm(t):
        t = t.merge(st, on="userId", how="left")
        t["z_score"] = (t["rating"] - t["train_mean"]) / t["train_std"]
        t["is_relevant"] = t["z_score"] > 0
        return t

    return norm(train), norm(val), norm(test)


def check_fold(df: pd.DataFrame, split: np.ndarray, k: int) -> dict:
    """Hard guarantees: one test + one val per evaluated user, and no train review is
    newer than that user's val/test review."""
    s = pd.Series(split, index=df.index)
    ev = df.assign(split=s)[s.isin(["train", "val", "test"])]
    per_user = pd.crosstab(ev["userId"], ev["split"])
    assert (per_user["test"] == 1).all() and (per_user["val"] == 1).all(), \
        f"fold {k}: every evaluated user must have exactly one val and one test review"
    t = ev.pivot_table(index="userId", columns="split", values="timestamp", aggfunc="max")
    assert (t["train"] <= t["val"]).all() and (t["val"] <= t["test"]).all(), \
        f"fold {k}: a train review is newer than val/test"
    return s.value_counts().to_dict()


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def _find_col(df, given, candidates, what):
    if given:
        return given
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"No {what} column found (tried {candidates}); pass --{what}_col")


def load_dataset(args) -> pd.DataFrame:
    base = os.path.join(args.prepared_dir, args.dataset_name)
    if os.path.exists(f"{base}.csv"):
        path = f"{base}.csv"
        raw = pd.read_csv(path)
    elif os.path.exists(f"{base}.parquet"):
        path = f"{base}.parquet"
        raw = pd.read_parquet(path)
    else:
        raise FileNotFoundError(f"Neither {base}.csv nor {base}.parquet exists")
    raw = raw.reset_index(drop=True)
    print(f"Loaded {path}: {len(raw):,} rows")

    user_col = _find_col(raw, args.user_col, USER_COL_CANDIDATES, "user")
    item_col = _find_col(raw, args.item_col, ITEM_COL_CANDIDATES, "item")
    time_col = _find_col(raw, args.time_col, TIME_COL_CANDIDATES, "time")

    df = pd.DataFrame({"review_idx": np.arange(len(raw))})       # == row_id in bertopic_extraction.py
    # same id encoding as bertopic_extraction.py: keep encoded ids, encode raw ones
    df["userId"] = raw[user_col].values if user_col == "userId" else LabelEncoder().fit_transform(raw[user_col])
    df["itemId"] = raw[item_col].values if item_col == "itemId" else LabelEncoder().fit_transform(raw[item_col])
    t = raw[time_col]
    df["timestamp"] = (t.astype("float64") if pd.api.types.is_numeric_dtype(t)
                       else pd.to_datetime(t, errors="coerce").astype("int64").astype("float64"))
    if args.rating_col in raw.columns:
        df["rating"] = raw[args.rating_col].values

    n_bad = int(df["timestamp"].isna().sum())
    if n_bad:
        print(f"  ! {n_bad:,} rows have an unparseable timestamp -> they sort first (oldest)")
        df["timestamp"] = df["timestamp"].fillna(-np.inf)
    print(f"  user={user_col!r}  item={item_col!r}  time={time_col!r}  "
          f"users={df['userId'].nunique():,}  items={df['itemId'].nunique():,}")
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Sliding-window LOO splits shared by recommender and topic profiles")
    p.add_argument("--dataset_name", required=True)
    p.add_argument("--prepared_dir", default="dataset")
    p.add_argument("--out_dir", default="splits")
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--user_col", default='userId')
    p.add_argument("--item_col", default='itemId')
    p.add_argument("--time_col", default='timestamp')
    p.add_argument("--rating_col", default="rating")
    return p.parse_args()


def main():
    args = parse_args()
    df = load_dataset(args)
    os.makedirs(args.out_dir, exist_ok=True)

    n_per_user = df.groupby("userId").size()
    kept = int((n_per_user >= args.n_folds + 2).sum())
    print(f"  Users with >= {args.n_folds + 2} interactions (evaluated in every fold): "
          f"{kept:,}/{len(n_per_user):,}")

    summary = []
    for k in range(args.n_folds):
        split = loo_labels(df, k, args.n_folds)
        counts = check_fold(df, split, k)
        path = os.path.join(args.out_dir, f"{args.dataset_name}_fold{k}.csv")
        pd.DataFrame({"row_id": df["review_idx"].values, "userId": df["userId"].values,
                      "itemId": df["itemId"].values, "split": split}).to_csv(path, index=False)
        summary.append({"fold": k, **{c: counts.get(c, 0) for c in ("train", "val", "test", "unused")}})
        print(f"  fold {k} -> {path}: {summary[-1]}")

    # test reviews must differ across folds (sliding window)
    tests = [set(pd.read_csv(os.path.join(args.out_dir, f"{args.dataset_name}_fold{k}.csv"))
                 .query("split == 'test'")["row_id"]) for k in range(args.n_folds)]
    assert all(not (tests[a] & tests[b]) for a in range(len(tests)) for b in range(a + 1, len(tests))), \
        "test sets overlap across folds"

    pd.DataFrame(summary).to_csv(os.path.join(args.out_dir, f"{args.dataset_name}_summary.csv"), index=False)
    print("Done. Test sets are disjoint across folds; all checks passed.")


if __name__ == "__main__":
    main()