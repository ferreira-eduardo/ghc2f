"""Final 5-fold experiment with the tuned hyperparameters, under the iRev baselines' protocol.

For every dataset / model / fold:
  * split      : splits/{dataset}_fold{k}.csv            (utils/leave_one_out_cv.py)
  * params     : grid_search/old_best_hyperparameters/{dataset}_{model}.csv
  * protocol   : utils/irev_protocol.py (same code as the iRev baselines)
      - positive = rating >= per-user threshold (train mean shrunk to the global mean,
        clipped to [0.6, 0.8] x max rating), computed on train only
      - ranking against the whole train catalog, user history excluded
        (train for val, train + val for test), ties at their mean position
      - graded nDCG@k over every held-out row (gain = rating / max rating);
        recall@k, MRR and beyond-accuracy over the positive rows only
  * train      : train split, early stopping on full-catalog val nDCG@10
  * test       : input = train + val history, ranked against the whole catalog
  * metrics    : nDCG / recall @ {3,5,10,15,20}, MRR, coverage, gini, entropy, ARP, APLT,
                 novelty @10, nDCG@10 by user activity and by head/tail target
  * output     : results/old_ghcf/{dataset}/

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
from model.cf_autoencoder import CFAutoEncoder
from model.ghc2f import GHC2F
from utils.catalog_eval import CatalogScorer, to_protocol_frame, user_text_arrays
from utils.dataset_utils import RankingTrainDataset, train_collate_fn, create_sparse_matrix
from utils.irev_protocol import ITEM_CHUNK, SELECTION, Protocol
from utils.train_model import train_model
from utils.utils import prepare_inputs, EarlyStoppingRanking

DATASETS = ["rotten_tomatoes"]
MODELS = ["ghc2f", "gated_hybrid"]
# Text-free AE_BPR has no tuned CSV of its own: it reuses the backbone of a text model
# (only BACKBONE_KEYS are read; the text/loss hyperparameters are ignored).
TEXTLESS_MODELS = {"AE_BPR": CFAutoEncoder}
HPARAM_SOURCES = ("gated_hybrid", "ghc2f")
MODEL_CLASSES = {**MODEL_REGISTRY, **TEXTLESS_MODELS}
FOLDS = [3, 4]
GHC2F_LOSS_KEYS = ("text_gamma", "contrastive_target", "stop_grad_cf", "cl_weight", "reg_weight")
BACKBONE_KEYS = ("layers", "nl_type", "dropout", "lr", "tied_weights")
INFO_COLS = ("fold", "dataset", "model", "text_features", "epochs_run", "train_time_s")


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


def load_best_params(hparams_dir, dataset, model_name, textless_source="gated_hybrid"):
    source = textless_source if model_name in TEXTLESS_MODELS else model_name
    path = os.path.join(hparams_dir, f"{dataset}_{source}.csv")
    if not os.path.exists(path):
        return None, path
    row = pd.read_csv(path).iloc[0].to_dict()
    params = {k: _py(row[k]) for k in BACKBONE_KEYS}
    loss_kwargs = {}
    if MODEL_CLASSES[model_name] is GHC2F:
        loss_kwargs = dict(STAGE1_GHC2F_DEFAULTS)
        loss_kwargs.update({k: _py(row[k]) for k in GHC2F_LOSS_KEYS if k in row and pd.notna(row[k])})
    return (params, loss_kwargs), path


def load_dataset(dataset, data_path=None):
    path = resolve_data_path(dataset, data_path)
    print(f"Loading {path} ...")
    raw = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    raw = raw.reset_index(drop=True)
    if "rating" not in raw.columns:
        raise ValueError(f"{path} has no rating column; the iRev protocol needs the original ratings")
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


def text_seen(df_text, history):
    """Reviews of `history` only, oldest -> newest (the per-user cap keeps the latest)."""
    order = (history.sort_values(["timestamp", "review_idx"], kind="mergesort").index
             if "timestamp" in history.columns else history.index)
    return df_text.loc[order]


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
    val = df[split == "val"]
    test = df[split == "test"]
    history = pd.concat([train, val])

    protocol = Protocol(*(to_protocol_frame(x) for x in (train, val, test)))
    print(f"  train={len(train):,}  {protocol.describe()}")

    # --- train ---
    train_matrix = create_sparse_matrix(train, total_users, total_items)
    text_train = text_seen(df_text, train)
    item_ids, item_text, item_mask = prepare_inputs(text_train, "itemId", text_cols, max_reviews=args.max_reviews)

    layer_sizes = build_layer_options(total_items)[params["layers"]](total_items)
    backbone = dict(layer_sizes=layer_sizes, nl_type=params["nl_type"], dp_drop_prob=params["dropout"],
                    learn_rate=params["lr"], tied_weights=params["tied_weights"])
    if model_name in TEXTLESS_MODELS:
        model = TEXTLESS_MODELS[model_name](**backbone).to(device)
    else:
        model = MODEL_CLASSES[model_name](
            num_users=total_users, num_items=total_items,
            text_dim=len(text_cols), text_latent_dim=64, **backbone, **loss_kwargs,
        ).to(device)
        model.set_item_corpus(item_ids.to(device), item_text.to(device), item_mask.to(device))

    train_dataset = RankingTrainDataset(train_matrix, text_train, train, seed=seed,
                                        max_user_reviews=args.max_reviews, mask_pos_in_input=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=train_collate_fn, num_workers=args.num_workers)

    # --- val: full catalog, train history as input and as exclusion ---
    val_scorer = CatalogScorer(model, train_matrix, train_dataset.user_text, len(text_cols),
                               protocol.items, args.eval_batch_size)
    val_history = []

    def validate(m):
        metrics = protocol.accuracy(val_scorer, "val", args.eval_chunk)
        val_history.append(metrics)
        print(f"  val ndcg@10={metrics['ndcg@10']:.4f}  recall@10={metrics['recall@10']:.4f}  "
              f"mrr={metrics['mrr']:.4f}")
        return metrics[SELECTION]

    early_stopping = EarlyStoppingRanking(patience=args.patience, verbose=True)
    t0 = time.time()
    best_val, train_losses = train_model(model, args.epochs, train_loader, None, early_stopping,
                                         validate=validate)
    early_stopping.load_best_into_model(model)
    train_time = time.time() - t0

    # --- test: train + val history as input and as exclusion ---
    hist_matrix = create_sparse_matrix(history, total_users, total_items)
    hist_text = user_text_arrays(text_seen(df_text, history), text_cols, args.max_reviews)
    test_scorer = CatalogScorer(model, hist_matrix, hist_text, len(text_cols),
                                protocol.items, args.eval_batch_size)
    metrics, rankings = protocol.report(test_scorer, "test", args.eval_chunk)

    rows = protocol.rows["test"]
    per_user = pd.DataFrame({"userId": rows["user"].to_numpy(), "itemId": rows["item"].to_numpy(),
                             "rating": rankings.ratings, "liked": rankings.liked,
                             "rank": rankings.ranks})  # 0-based, ties at mean position

    row = {"fold": fold, **metrics, f"best_val_{SELECTION}": float(best_val),
           "epochs_run": len(train_losses), "train_time_s": round(train_time, 1)}
    losses = {"fold": fold, f"best_val_{SELECTION}": float(best_val),
              "train_losses": json.dumps([float(x) for x in train_losses]),
              "val_metrics": json.dumps(val_history)}

    del model, early_stopping, train_loader, train_dataset, val_scorer, test_scorer
    del item_ids, item_text, item_mask
    gc.collect()
    torch.cuda.empty_cache()
    return row, losses, per_user


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def summarize(df_folds):
    metric_cols = [c for c in df_folds.columns
                   if c not in INFO_COLS and pd.api.types.is_numeric_dtype(df_folds[c])]
    return pd.DataFrame({"metric": metric_cols,
                         "mean": [df_folds[c].mean() for c in metric_cols],
                         "std": [df_folds[c].std(ddof=1) if len(df_folds) > 1 else 0.0 for c in metric_cols],
                         "n_folds": len(df_folds)})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=DATASETS)
    p.add_argument("--models", nargs="+", default=MODELS, choices=sorted(MODEL_CLASSES))
    p.add_argument("--folds", nargs="+", type=int, default=FOLDS)
    p.add_argument("--splits_dir", default="splits")
    p.add_argument("--hparams_dir", default="grid_search/old_best_hyperparameters")
    p.add_argument("--textless_hparams_from", default="gated_hybrid", choices=HPARAM_SOURCES,
                   help="Which tuned text model's backbone (layers/nl_type/dropout/lr/tied_weights) "
                        "the text-free models (AE_BPR) reuse")
    p.add_argument("--set", nargs="+", default=[], metavar="KEY=VALUE",
                   help="Override tuned hyperparameters, e.g. --set cl_weight=0 (ablations/diagnostics)")
    p.add_argument("--out_dir", default="results/new_ghc2f")
    p.add_argument("--data_path", default=None, help="Default: same lookup as optimizer_hyper.py")
    p.add_argument("--embeddings_path", default=None,
                   help="Default: same lookup as optimizer_hyper.py. May use {dataset} and {fold}, e.g. "
                        "results/topics/{dataset}/fold{fold}/review_topic_dist.npy")
    p.add_argument("--epochs", type=int, default=STAGE3_EPOCHS)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--eval_batch_size", type=int, default=512, help="Users per model forward at evaluation")
    p.add_argument("--eval_chunk", type=int, default=ITEM_CHUNK,
                   help="Rows ranked per pass (each holds a rows x catalog score matrix on the GPU)")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--max_reviews", type=int, default=MAX_REVIEWS)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true", help="Re-run folds that already have results")
    args = p.parse_args()

    overrides = {}
    for kv in args.set:
        k, v = kv.split("=", 1)
        if k not in BACKBONE_KEYS + GHC2F_LOSS_KEYS:
            raise SystemExit(f"--set: unknown key {k!r}; allowed {BACKBONE_KEYS + GHC2F_LOSS_KEYS}")
        try:
            v = float(v) if any(c in v for c in ".eE") else int(v)
        except ValueError:
            v = _py(v)
        overrides[k] = v

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for dataset in args.datasets:
        todo = {}
        for m in args.models:
            best, path = load_best_params(args.hparams_dir, dataset, m, args.textless_hparams_from)
            if best is None:
                print(f"[skip] {dataset}/{m}: no tuned hyperparameters at {path}")
            else:
                params, loss_kwargs = best
                for k, v in overrides.items():
                    target = params if k in BACKBONE_KEYS else loss_kwargs
                    if target is loss_kwargs and MODEL_CLASSES[m] is not GHC2F:
                        continue
                    target[k] = v
                todo[m] = best
                print(f"[hparams] {dataset}/{m}: {path}")
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
            if prev is not None and "ndcg@10" not in prev.columns:
                raise SystemExit(f"{folds_csv} was made with the old sampled-negatives protocol; "
                                 f"move it away or pass --overwrite")
            rows = prev.to_dict("records") if prev is not None else []
            loss_rows = (pd.read_csv(losses_csv).to_dict("records")
                         if prev is not None and os.path.exists(losses_csv) else [])

            with open(os.path.join(out_dir, f"{model_name}_config.json"), "w") as f:
                json.dump({"dataset": dataset, "model": model_name, "data_path": data_path,
                           "protocol": "irev_full_catalog", "selection": SELECTION,
                           "backbone": params, "loss_kwargs": loss_kwargs, "overrides": overrides,
                           "hparams_from": (args.textless_hparams_from if model_name in TEXTLESS_MODELS
                                            else model_name),
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
                print("  TEST " + "  ".join(f"{k}={row[k]:.4f}" for k in
                                             ("ndcg@10", "recall@10", "mrr", "coverage@10", "aplt@10")))

                # saved after every fold, so a crash keeps the finished folds
                df_folds = pd.DataFrame(rows).sort_values("fold")
                df_folds.to_csv(folds_csv, index=False)
                pd.DataFrame(loss_rows).to_csv(losses_csv, index=False)
                per_user.to_csv(os.path.join(out_dir, f"{model_name}_fold{fold}_ranks_rt.csv"), index=False)
                summarize(df_folds).to_csv(os.path.join(out_dir, f"{model_name}_summary_rt.csv"), index=False)

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
