"""Final 5-fold experiment with the tuned hyperparameters.

For every dataset / model / fold:
  * split      : splits/{dataset}_fold{k}.csv            (utils/leave_one_out_cv.py)
  * params     : grid_search/best_hyperparameters/{dataset}_{model}.csv
  * train      : train split, early stopping on val HR@10 (same 99 fixed negatives every epoch)
  * test       : input = train + val history, 1 held-out test item vs 99 fixed unseen negatives
  * metrics    : HR@{3,5,10,15}, NDCG@{3,5,10,15}, MRR
  * output     : results/new_ghcf/{dataset}/

Model construction, layer grid, text inputs and review caps are the same as in
grid_search/optimizer_hyper.py, so the final runs match what was tuned.

Usage:
  python run_experiment.py                                   # all datasets / models / folds
  python run_experiment.py --datasets rotten_tomatoes --models ghc2f --folds 0 1
"""
import argparse
import gc
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from grid_search.optimizer_hyper import (MODEL_REGISTRY, STAGE1_GHC2F_DEFAULTS, STAGE3_EPOCHS, MAX_REVIEWS,
                                         build_layer_options, resolve_data_path, resolve_embeddings_path)
from model.ghc2f import GHC2F
from utils.dataset_utils import (RankingTrainDataset, train_collate_fn, loocv_collate_fn, create_sparse_matrix,
                                 build_interacted_by_user, build_fixed_negatives)
from utils.train_model import train_model
from utils.utils import prepare_inputs, EarlyStoppingRanking

DATASETS = ["imdb", "rotten_tomatoes", "amazon"]
MODELS = ["gated_hybrid", "ghc2f"]
FOLDS = [0, 1, 2, 3, 4]
KS = [3, 5, 10, 15]
GHC2F_LOSS_KEYS = ("text_gamma", "contrastive_target", "stop_grad_cf", "cl_weight", "reg_weight")
BACKBONE_KEYS = ("layers", "nl_type", "dropout", "lr", "tied_weights")


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def _py(v):
    """numpy scalar / 'True' string -> plain python value."""
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, str) and v in ("True", "False"):
        return v == "True"
    return v


def load_best_params(hparams_dir, dataset, model_name):
    path = os.path.join(hparams_dir, f"{dataset}_{model_name}.csv")
    if not os.path.exists(path):
        return None, path
    row = pd.read_csv(path).iloc[0].to_dict()
    params = {k: _py(row[k]) for k in BACKBONE_KEYS}
    loss_kwargs = {}
    if MODEL_REGISTRY[model_name] is GHC2F:
        loss_kwargs = dict(STAGE1_GHC2F_DEFAULTS)
        loss_kwargs.update({k: _py(row[k]) for k in GHC2F_LOSS_KEYS if k in row and pd.notna(row[k])})
    return (params, loss_kwargs), path


def load_dataset(dataset, data_path=None):
    path = resolve_data_path(dataset, data_path)
    print(f"Loading {path} ...")
    raw = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    raw = raw.reset_index(drop=True)
    cols = [c for c in ("userId", "itemId", "rating", "timestamp") if c in raw.columns]
    df = raw[cols].copy()
    del raw
    df["userId"] = df["userId"].astype("int64")
    df["itemId"] = df["itemId"].astype("int64")
    df["review_idx"] = np.arange(len(df))
    return df, path


def load_split(df, split_path):
    split = pd.read_csv(split_path)
    if not (len(split) == len(df)
            and (split["row_id"].values == df["review_idx"].values).all()
            and (split["userId"].values == df["userId"].values).all()
            and (split["itemId"].values == df["itemId"].values).all()):
        raise ValueError(f"{split_path} is not row-aligned with the dataset")
    return split["split"].values


def load_text(df, emb_path):
    emb = np.load(emb_path).astype(np.float32, copy=False)
    if len(emb) != len(df):
        raise ValueError(f"{emb_path}: {len(emb)} rows != {len(df)} data rows")
    df_text = pd.DataFrame(emb)
    df_text["userId"] = df["userId"].values
    df_text["itemId"] = df["itemId"].values
    text_cols = [c for c in df_text.columns if str(c).isdigit()]
    return df_text, text_cols


def drop_rereviews(history, eval_df, what):
    """An eval item already in the user's input history is not a ranking task."""
    seen = pd.MultiIndex.from_frame(history[["userId", "itemId"]])
    mask = pd.MultiIndex.from_frame(eval_df[["userId", "itemId"]]).isin(seen)
    if mask.any():
        print(f"  Dropping {int(mask.sum())} {what} rows whose item is already in the user's history")
    return eval_df[~mask]


def text_seen(df_text, history):
    """Reviews of `history` only, oldest -> newest (the per-user cap keeps the latest)."""
    order = (history.sort_values(["timestamp", "review_idx"], kind="mergesort").index
             if "timestamp" in history.columns else history.index)
    return df_text.loc[order]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_ranking(model, loader, ks=KS):
    """Rank of the held-out item among [pos + 99 negatives]; ties count against it."""
    model.eval()
    user_ids, pos_ids, ranks = [], [], []
    for batch in loader:
        batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        candidates = torch.cat([batch["pos_item_id"].unsqueeze(-1), batch["neg_item_id"]], dim=1)
        scores = model(batch, item_indices=candidates).recon
        r = (scores[:, 1:] >= scores[:, :1]).sum(dim=1) + 1
        ranks.append(r.cpu().numpy())
        user_ids.append(batch["user_ids"].cpu().numpy())
        pos_ids.append(batch["pos_item_id"].cpu().numpy())

    ranks = np.concatenate(ranks).astype(np.float64)
    metrics = {}
    for k in ks:
        metrics[f"HR@{k}"] = float(np.mean(ranks <= k))
    for k in ks:
        metrics[f"NDCG@{k}"] = float(np.mean(np.where(ranks <= k, 1.0 / np.log2(ranks + 1), 0.0)))
    metrics["MRR"] = float(np.mean(1.0 / ranks))
    metrics["n_users"] = int(len(ranks))
    per_user = pd.DataFrame({"userId": np.concatenate(user_ids), "itemId": np.concatenate(pos_ids),
                             "rank": ranks.astype(int)})
    return metrics, per_user


# ---------------------------------------------------------------------------
# One fold
# ---------------------------------------------------------------------------
def run_fold(args, model_name, params, loss_kwargs, df, split, df_text, text_cols, fold, device):
    seed = args.seed + fold
    np.random.seed(seed)
    torch.manual_seed(seed)

    total_items = int(df.itemId.max() + 1)
    total_users = int(df.userId.max() + 1)

    train = df[split == "train"]
    val = drop_rereviews(train, df[split == "val"], "val")
    history = pd.concat([train, df[split == "val"]])
    test = drop_rereviews(history, df[split == "test"], "test")
    if args.only_relevant_test:
        if "rating" not in df.columns:
            raise ValueError("--only_relevant_test needs a rating column")
        stats = train.groupby("userId")["rating"].agg(["mean", "std"])
        std = stats["std"].replace(0, 1.0).fillna(1.0)
        z = (test["rating"].values - stats["mean"].reindex(test["userId"]).values) / std.reindex(test["userId"]).values
        test = test[z > 0]
    print(f"  train={len(train):,}  val users={len(val):,}  test users={len(test):,}")

    # --- train / val (identical to the tuning setup) ---
    train_matrix = create_sparse_matrix(train, total_users, total_items)
    text_train = text_seen(df_text, train)
    inter_train = build_interacted_by_user(train)
    val_negs = build_fixed_negatives(val, inter_train, total_items, seed=seed)

    item_ids, item_text, item_mask = prepare_inputs(text_train, "itemId", text_cols, max_reviews=args.max_reviews)

    layer_sizes = build_layer_options(total_items)[params["layers"]](total_items)
    model = MODEL_REGISTRY[model_name](
        layer_sizes=layer_sizes, num_users=total_users, num_items=total_items,
        text_dim=len(text_cols), text_latent_dim=64,
        nl_type=params["nl_type"], dp_drop_prob=params["dropout"], learn_rate=params["lr"],
        tied_weights=params["tied_weights"], **loss_kwargs,
    ).to(device)
    model.set_item_corpus(item_ids.to(device), item_text.to(device), item_mask.to(device))

    train_loader = DataLoader(
        RankingTrainDataset(train_matrix, text_train, train, seed=seed,
                            max_user_reviews=args.max_reviews, mask_pos_in_input=True),
        batch_size=args.batch_size, shuffle=True, collate_fn=train_collate_fn, num_workers=args.num_workers)
    val_loader = DataLoader(
        RankingTrainDataset(train_matrix, text_train, val, max_user_reviews=args.max_reviews,
                            mask_pos_in_input=False),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        collate_fn=lambda b: loocv_collate_fn(b, inter_train, total_items, fixed_negatives=val_negs))

    early_stopping = EarlyStoppingRanking(patience=args.patience, verbose=True)
    t0 = time.time()
    best_val_hr, train_losses = train_model(model, args.epochs, train_loader, val_loader, early_stopping)
    early_stopping.load_best_into_model(model)
    train_time = time.time() - t0

    # --- test: the user's train + val history is the input ---
    hist_matrix = create_sparse_matrix(history, total_users, total_items)
    text_hist = text_seen(df_text, history)
    inter_hist = build_interacted_by_user(history)
    test_negs = build_fixed_negatives(test, inter_hist, total_items, seed=seed + 1000)
    test_loader = DataLoader(
        RankingTrainDataset(hist_matrix, text_hist, test, max_user_reviews=args.max_reviews,
                            mask_pos_in_input=False),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        collate_fn=lambda b: loocv_collate_fn(b, inter_hist, total_items, fixed_negatives=test_negs))
    metrics, per_user = evaluate_ranking(model, test_loader)

    row = {"fold": fold, **metrics, "best_val_HR@10": float(best_val_hr),
           "epochs_run": len(train_losses), "train_time_s": round(train_time, 1)}
    losses = {"fold": fold, "best_val_HR@10": float(best_val_hr), "train_losses": json.dumps(
        [float(x) for x in train_losses])}

    del model, early_stopping, train_loader, val_loader, test_loader, item_ids, item_text, item_mask
    gc.collect()
    torch.cuda.empty_cache()
    return row, losses, per_user


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def summarize(df_folds):
    metric_cols = [f"HR@{k}" for k in KS] + [f"NDCG@{k}" for k in KS] + ["MRR"]
    return pd.DataFrame({"metric": metric_cols,
                         "mean": [df_folds[c].mean() for c in metric_cols],
                         "std": [df_folds[c].std(ddof=1) if len(df_folds) > 1 else 0.0 for c in metric_cols],
                         "n_folds": len(df_folds)})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=DATASETS)
    p.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    p.add_argument("--folds", nargs="+", type=int, default=FOLDS)
    p.add_argument("--splits_dir", default="splits")
    p.add_argument("--hparams_dir", default="grid_search/best_hyperparameters")
    p.add_argument("--out_dir", default="results/new_ghcf")
    p.add_argument("--data_path", default=None, help="Default: same lookup as optimizer_hyper.py")
    p.add_argument("--embeddings_path", default=None,
                   help="Default: same lookup as optimizer_hyper.py. May use {dataset} and {fold}, e.g. "
                        "results/topics/{dataset}/fold{fold}/review_topic_dist.npy")
    p.add_argument("--epochs", type=int, default=STAGE3_EPOCHS)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--max_reviews", type=int, default=MAX_REVIEWS)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--only_relevant_test", action="store_true",
                   help="Keep only test items rated above the user's train mean (old run_experiment.py behaviour)")
    p.add_argument("--overwrite", action="store_true", help="Re-run folds that already have results")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for dataset in args.datasets:
        todo = {}
        for m in args.models:
            best, path = load_best_params(args.hparams_dir, dataset, m)
            if best is None:
                print(f"[skip] {dataset}/{m}: no tuned hyperparameters at {path}")
            else:
                todo[m] = best
        if not todo:
            continue
        missing = [f for f in (os.path.join(args.splits_dir, f"{dataset}_fold{k}.csv") for k in args.folds)
                   if not os.path.exists(f)]
        if missing:
            print(f"[skip] {dataset}: missing split files {missing}")
            continue

        out_dir = os.path.join(args.out_dir, dataset)
        os.makedirs(out_dir, exist_ok=True)
        df, data_path = load_dataset(dataset, args.data_path)

        emb_template = args.embeddings_path
        per_fold_text = emb_template is not None and "{fold}" in emb_template
        cached_text = None
        if not per_fold_text:
            emb_path = (emb_template.format(dataset=dataset) if emb_template
                        else resolve_embeddings_path(dataset))
            print(f"Loading text features {emb_path} ...")
            cached_text = (*load_text(df, emb_path), emb_path)

        for model_name, (params, loss_kwargs) in todo.items():
            folds_csv = os.path.join(out_dir, f"{model_name}_folds.csv")
            losses_csv = os.path.join(out_dir, f"{model_name}_losses.csv")
            prev = pd.read_csv(folds_csv) if os.path.exists(folds_csv) and not args.overwrite else None
            rows = prev.to_dict("records") if prev is not None else []
            loss_rows = (pd.read_csv(losses_csv).to_dict("records")
                         if prev is not None and os.path.exists(losses_csv) else [])

            with open(os.path.join(out_dir, f"{model_name}_config.json"), "w") as f:
                json.dump({"dataset": dataset, "model": model_name, "data_path": data_path,
                           "backbone": params, "loss_kwargs": loss_kwargs,
                           "args": vars(args)}, f, indent=2, default=str)

            for fold in args.folds:
                if any(int(r["fold"]) == fold for r in rows):
                    print(f"[done] {dataset}/{model_name} fold {fold} (use --overwrite to re-run)")
                    continue
                print("#" * 60)
                print(f"{dataset} | {model_name} | fold {fold} | {params} {loss_kwargs}")
                print("#" * 60)

                split = load_split(df, os.path.join(args.splits_dir, f"{dataset}_fold{fold}.csv"))
                if per_fold_text:
                    emb_path = emb_template.format(dataset=dataset, fold=fold)
                    print(f"Loading text features {emb_path} ...")
                    df_text, text_cols = load_text(df, emb_path)
                else:
                    df_text, text_cols, emb_path = cached_text

                row, losses, per_user = run_fold(args, model_name, params, loss_kwargs,
                                                 df, split, df_text, text_cols, fold, device)
                row.update(dataset=dataset, model=model_name, text_features=emb_path)
                rows.append(row)
                loss_rows.append(losses)
                print("  TEST " + "  ".join(f"{k}={v:.4f}" for k, v in row.items()
                                             if k.startswith(("HR@", "NDCG@", "MRR"))))

                # saved after every fold, so a crash keeps the finished folds
                df_folds = pd.DataFrame(rows).sort_values("fold")
                df_folds.to_csv(folds_csv, index=False)
                pd.DataFrame(loss_rows).to_csv(losses_csv, index=False)
                per_user.to_csv(os.path.join(out_dir, f"{model_name}_fold{fold}_ranks.csv"), index=False)
                summarize(df_folds).to_csv(os.path.join(out_dir, f"{model_name}_summary.csv"), index=False)

                if per_fold_text:
                    del df_text
                    gc.collect()

            if rows:
                print(f"\n{dataset}/{model_name} summary over {len(rows)} folds:")
                print(summarize(pd.DataFrame(rows)).to_string(index=False))

        del df, cached_text
        gc.collect()


if __name__ == "__main__":
    main()
