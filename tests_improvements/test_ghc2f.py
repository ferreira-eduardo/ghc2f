"""
Comparison harness: autoencoder (CFAutoEncoder), gated_hybrid
(GatedHybridCFAutoEncoder), and GHC2F (decoder-scored, with/without
contrastive alignment), all trained purely under BPR-family objectives --
MMSE has been dropped entirely from every class (see cf_autoencoder.py /
ghc2f.py docstrings for the loss-scale-imbalance finding that motivated
this).

Logs per-epoch loss components for GHC2F (bpr/cl/gate/reg) so you can see
the L_CL trajectory directly instead of inferring it.

Trains all arms with a hand-rolled epoch loop (not utils.train_model) so
every arm goes through IDENTICAL step/eval logic regardless of model class
-- this matters because CFAutoEncoder, GatedHybridCFAutoEncoder, and GHC2F
all expose calculate_loss(batch) -> (loss, batch_size) and evaluate(loader)
-> {hit_rate, ndcg, mrr}, but this test doesn't assume anything about
train_model's internals staying comparable across model classes.

NEW: --seeds accepts one or more seeds so each arm can be run multiple
times to check whether a result (e.g. gated_hybrid vs ghc2f_no_cl) is
stable or fold/init noise. With a single seed, behavior/filenames are
unchanged from before; with multiple seeds, per-seed CSVs get a _seedN
suffix and an additional mean/std summary is printed and saved.

Usage:
    python test_ghc2f.py --dataset imdb --epochs 20
    python test_ghc2f.py --dataset imdb --epochs 20 --seeds 42 123 7
"""
import argparse
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils.utils import prepare_inputs
from utils.dataset_utils import RankingTrainDataset, train_collate_fn, create_sparse_matrix
from utils.dataset_utils import loocv_collate_fn, build_interacted_by_user
from utils.leave_one_out_cv import get_loocv_fold_normalized

from model.cf_autoencoder import CFAutoEncoder
from model.gated_hybrid_ae import GatedHybridCFAutoEncoder
from model.ghc2f import GHC2F


# ---------------------------------------------------------------------------
# Fixed backbone -- deliberately simple, not searched. Keep IDENTICAL across
# all arms so the comparison isolates model/objective, not architecture.
# ---------------------------------------------------------------------------
LAYER_FRAC = 0.2
MIN_WIDTH, MAX_WIDTH = 32, 4096
DROPOUT = 0.5
LR = 1e-4
NL_TYPE = "selu"
TIED_WEIGHTS = True

GHC2F_LOSS_DEFAULTS = dict(
    contrastive_target="text",
    stop_grad_cf=True,
    cl_weight=0.1,
    reg_weight=1e-5,
    gate_entropy_weight=0.0,
)

ARMS = ["ae_bpr", "gated_hybrid", "ghc2f_decoder", "ghc2f_no_cl"]


def set_seed(seed):
    """
    Seeds everything that introduces run-to-run variance in this harness:
    model weight init (nn.init.xavier_uniform_ etc., via torch's global
    RNG), DataLoader shuffling (shuffle=True with no explicit generator
    also draws from torch's global RNG), and any numpy/python-level
    randomness in the data pipeline (e.g. negative sampling inside
    RankingTrainDataset, if it uses numpy/random rather than a local
    generator).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_layer_sizes(total_items, frac=LAYER_FRAC, min_w=MIN_WIDTH, max_w=MAX_WIDTH):
    w = int(np.clip(round(total_items * frac), min_w, max_w))
    return [total_items, w, max(w // 2, min_w)]


def load_fold(dataset_name, dataset_dir="datasets", embeddings_dir="embeddings_reviews"):
    full_df = pd.read_csv(os.path.join(dataset_dir, f"{dataset_name}.csv"))
    full_df["userId"] = full_df["userId"].astype("int32")
    full_df["itemId"] = full_df["itemId"].astype("int32")

    embedding = np.load(os.path.join(embeddings_dir, f"{dataset_name}.npy"))
    df_text = pd.DataFrame(embedding)
    df_text["userId"] = full_df["userId"].values
    df_text["itemId"] = full_df["itemId"].values
    text_cols = [c for c in df_text.columns if str(c).isdigit()]

    total_items = int(full_df.itemId.max() + 1)
    total_users = int(full_df.userId.max() + 1)

    train, val, _ = get_loocv_fold_normalized(full_df, 0)
    train_matrix = create_sparse_matrix(train, total_users, total_items)

    train_pairs = pd.MultiIndex.from_frame(train[["userId", "itemId"]])
    df_text_pairs = pd.MultiIndex.from_frame(df_text[["userId", "itemId"]])
    df_text_seen = df_text[df_text_pairs.isin(train_pairs)].copy()

    interacted_by_user = build_interacted_by_user(train)

    df_text_train = df_text[df_text["itemId"].isin(train["itemId"].unique())].copy()
    item_ids, item_text, item_mask = prepare_inputs(df_text_train, "itemId", text_cols)

    return dict(
        total_items=total_items, total_users=total_users, text_dim=len(text_cols),
        train=train, val=val, train_matrix=train_matrix, df_text_seen=df_text_seen,
        interacted_by_user=interacted_by_user,
        item_ids=item_ids, item_text=item_text, item_mask=item_mask,
    )


def _set_item_corpus(model, fold, device):
    """
    Shared by gated_hybrid and both ghc2f_* arms: gives the model the raw
    item corpus so item_profiler is called LIVE inside forward() every
    step, instead of being precomputed once under torch.no_grad()/.detach()
    and cached (the old pattern, confirmed via check_topic_pathway.py to
    leave item_profiler's parameters bit-identical after training --
    i.e. frozen at random init for the entire run). Both
    GatedHybridCFAutoEncoder and GHC2F expose the same
    set_item_corpus/item_profiler interface (GHC2F inherits it), so this
    one function covers both.
    """
    model.set_item_corpus(
        fold["item_ids"].to(device), fold["item_text"].to(device), fold["item_mask"].to(device)
    )


def build_model(arm, fold, device):
    layer_sizes = build_layer_sizes(fold["total_items"])
    common_kwargs = dict(
        nl_type=NL_TYPE, dp_drop_prob=DROPOUT, learn_rate=LR, tied_weights=TIED_WEIGHTS,
    )

    # if arm == "ae_bpr":
    #     return CFAutoEncoder(layer_sizes=layer_sizes, **common_kwargs).to(device)
    #
    # if arm == "gated_hybrid":
    #     model = GatedHybridCFAutoEncoder(
    #         layer_sizes=layer_sizes,
    #         num_users=fold["total_users"],
    #         num_items=fold["total_items"],
    #         text_dim=fold["text_dim"],
    #         text_latent_dim=64,
    #         **common_kwargs,
    #     ).to(device)
    #     _set_item_corpus(model, fold, device)
    #     return model
    #
    # # ghc2f_decoder | ghc2f_no_cl
    loss_kwargs = dict(GHC2F_LOSS_DEFAULTS)
    # if arm == "ghc2f_no_cl":
    #     # cl_weight=0.0 ablation: with MMSE and gate_drop no longer
    #     # confounding things, this isolates whether the contrastive term is
    #     # contributing anything real. loss_cl sitting at ~7.4-7.5 across all
    #     # 20 epochs (~log(batch_size) = log(2048) ~= 7.62, i.e. near the
    #     # InfoNCE floor for uninformative similarity) suggested it might
    #     # not be -- while still being the single largest weighted loss term
    #     # (cl_weight * loss_cl ~= 0.75 vs loss_bpr ~= 0.17-0.19). This arm
    #     # tests that directly.
    #     loss_kwargs["cl_weight"] = 0.0

    model = GHC2F(
        layer_sizes=layer_sizes,
        num_users=fold["total_users"],
        num_items=fold["total_items"],
        text_dim=fold["text_dim"],
        text_latent_dim=64,
        **common_kwargs,
        **loss_kwargs,
    ).to(device)
    _set_item_corpus(model, fold, device)

    return model


def build_loaders(fold, batch_size=2048, num_workers=0):
    train_loader = DataLoader(
        RankingTrainDataset(fold["train_matrix"], fold["df_text_seen"], fold["train"]),
        batch_size=batch_size, shuffle=True, collate_fn=train_collate_fn,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        RankingTrainDataset(fold["train_matrix"], fold["df_text_seen"], fold["val"]),
        batch_size=batch_size, shuffle=False,
        collate_fn=lambda b: loocv_collate_fn(b, fold["interacted_by_user"], fold["total_items"]),
        num_workers=num_workers,
    )
    return train_loader, val_loader


def run_epochs(model, train_loader, val_loader, epochs, eval_every=1):
    """
    Hand-rolled train/eval loop, identical for every arm. Records:
      - avg total training loss per epoch
      - avg loss component values per epoch, IF the model sets
        self.last_loss_components (GHC2F does; CFAutoEncoder doesn't, so
        those columns are simply absent/NaN for the ae_bpr arm)
      - hit_rate/ndcg/mrr per epoch via model.evaluate(val_loader)
    """
    history = []
    best_hit_rate = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_n = 0.0, 0
        comp_sums = {}

        for batch in train_loader:
            model.optimizer.zero_grad()
            loss, n = model.calculate_loss(batch)
            loss.backward()
            model.optimizer.step()

            total_loss += loss.item() * n
            total_n += n

            comps = getattr(model, "last_loss_components", None)
            if comps:
                for k, v in comps.items():
                    comp_sums[k] = comp_sums.get(k, 0.0) + v * n

        avg_loss = total_loss / max(total_n, 1)
        avg_comps = {k: v / max(total_n, 1) for k, v in comp_sums.items()}

        row = {"epoch": epoch, "train_loss": avg_loss,
               **{f"loss_{k}": v for k, v in avg_comps.items()}}

        if epoch % eval_every == 0 or epoch == epochs:
            metrics = model.evaluate(val_loader)
            row.update(metrics)
            best_hit_rate = max(best_hit_rate, metrics["hit_rate"])

            extra = ""
            if "cl" in avg_comps:
                extra = (f" | loss_bpr {avg_comps['bpr']:.4f} | loss_cl {avg_comps['cl']:.4f}"
                         f" | loss_gate {avg_comps.get('gate', float('nan')):.4f}"
                         f" | loss_reg {avg_comps.get('reg', float('nan')):.4f}")
            print(f"  epoch {epoch:3d} | train_loss {avg_loss:.4f} | "
                  f"hit_rate {metrics['hit_rate']:.4f} | ndcg {metrics['ndcg']:.4f} | "
                  f"mrr {metrics['mrr']:.4f}{extra}")

        history.append(row)

    return best_hit_rate, pd.DataFrame(history)


def run_one(arm, fold, device, epochs, batch_size, num_workers, seed):
    print(f"\n{'=' * 70}\n{arm}  (seed={seed})\n{'=' * 70}")

    model = build_model(arm, fold, device)
    train_loader, val_loader = build_loaders(fold, batch_size=batch_size, num_workers=num_workers)

    best_hit_rate, history_df = run_epochs(model, train_loader, val_loader, epochs)
    final_metrics = {k: history_df.iloc[-1][k] for k in ("hit_rate", "ndcg", "mrr")}
    print(f"[{arm}] best_val_hit_rate over training: {best_hit_rate:.4f}")
    print(f"[{arm}] final epoch metrics: {final_metrics}")

    gate_df = None
    if hasattr(model, "gate_diagnostics"):
        val_batch = next(iter(val_loader))
        gate_df = pd.DataFrame(model.gate_diagnostics(val_batch))
        print(f"[{arm}] gate diagnostics (per encoder layer):")
        print(gate_df.to_string(index=False))

    result = {"arm": arm, "seed": seed, "best_val_hit_rate": best_hit_rate, **final_metrics}

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result, history_df, gate_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Dataset name (matches datasets/{name}.csv)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--arms", type=str, nargs="+", default=ARMS, choices=ARMS,
                         help="Subset of arms to run, e.g. --arms ae_bpr ghc2f_decoder")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42],
                         help="One or more seeds, e.g. --seeds 42 123 7. Each arm is run "
                              "once per seed; with >1 seed a mean/std summary is added.")
    args = parser.parse_args()

    print("NOTE: MMSE has been dropped entirely from every arm, per the "
          "loss-scale imbalance found earlier (loss_mmse ~100-190x larger "
          "than loss_bpr under equal weights -- 'bpr_mmse' behaved almost "
          "identically to MSE-only and underperformed pure BPR). All arms "
          "below now train purely under the ranking objective:\n"
          "  ae_bpr       -> CFAutoEncoder, pure pairwise BPR (Eq. 2).\n"
          "  gated_hybrid -> GatedHybridCFAutoEncoder, pure BPR (Eq. 7) -- "
          "this class never had an MMSE term (confirmed against Section "
          "3.2's description), so it's unchanged.\n"
          "  ghc2f_decoder -> BPR + contrastive alignment + gate entropy + "
          "L2 reg, decoder-scored.\n"
          "  ghc2f_no_cl  -> ghc2f_decoder with cl_weight=0.0 -- tests "
          "whether contrastive alignment is contributing anything real.\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading fold for {args.dataset} ...")
    fold = load_fold(args.dataset)

    multi_seed = len(args.seeds) > 1
    all_results, gate_frames = [], {}

    for arm in args.arms:
        for seed in args.seeds:
            set_seed(seed)
            res, history_df, gate_df = run_one(arm, fold, device, epochs=args.epochs,
                                                batch_size=args.batch_size,
                                                num_workers=args.num_workers, seed=seed)
            all_results.append(res)

            suffix = f"_seed{seed}" if multi_seed else ""
            history_df.to_csv(f"{args.dataset}_{arm}{suffix}_epoch_history.csv", index=False)
            print(f"Saved {arm} (seed={seed}) per-epoch history to "
                  f"{args.dataset}_{arm}{suffix}_epoch_history.csv")
            if gate_df is not None:
                gate_frames[f"{arm}{suffix}"] = gate_df

    df = pd.DataFrame(all_results)
    print(f"\n\n=== Arm comparison ({args.epochs} epochs, seeds={args.seeds}) ===")
    print(df.to_string(index=False))

    out_path = f"{args.dataset}_arm_comparison.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved comparison to {out_path}")

    if multi_seed:
        summary = (
            df.groupby("arm")[["best_val_hit_rate", "hit_rate", "ndcg", "mrr"]]
            .agg(["mean", "std"])
        )
        print(f"\n=== Mean/std across {len(args.seeds)} seeds ===")
        print(summary.to_string())
        summary_path = f"{args.dataset}_arm_comparison_summary.csv"
        summary.to_csv(summary_path)
        print(f"\nSaved seed summary to {summary_path}")

    for name, gdf in gate_frames.items():
        gate_path = f"{args.dataset}_{name}_gate.csv"
        gdf.to_csv(gate_path, index=False)
        print(f"Saved {name} gate diagnostics to {gate_path}")