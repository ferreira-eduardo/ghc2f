"""
Comparison harness: AE_BPR (CFAutoEncoder, pure collaborative backbone) vs.
SemanticDisentangledAE (adds the sentiment-alignment pathway on top of the
same backbone). Isolates the question this script exists to answer: does the
semantic pathway change ranking-quality metrics (ndcg/mrr) and/or
beyond-accuracy metrics (novelty/diversity/serendipity) without changing
hit_rate -- see prior debugging notes on why hit_rate staying flat while
ndcg/mrr move is the pattern worth checking for here.

Trimmed from the larger GHC2F/gated-hybrid comparison harness: this script
only builds ae_bpr and semantic_ae (+ semantic_ae_detach as an optional null
baseline), so all gated-hybrid/GHC2F/AE_Contrastive/text-item-corpus/gate-
diagnostics machinery has been removed. If you need those arms, use the
original harness.

Beyond-accuracy metrics (novelty/diversity/serendipity) are EXPENSIVE (a full
item-catalog scoring pass per batch, on top of the sampled-candidate pass
already used for hit_rate/ndcg/mrr) -- see CFAutoEncoder.evaluate(). They are
therefore computed ONLY ONCE per (arm, seed), on the model's final-epoch
state, after the training loop finishes -- never per epoch. This is a
documented design choice, not an oversight: if you need them at intermediate
epochs too, that's a separate, explicit decision to make (and pay for).

NOTE on "final-epoch state": this script evaluates beyond-accuracy metrics on
whatever the model looks like after the LAST training epoch, not on a saved
best-val-hit-rate checkpoint (best_val_hit_rate is still tracked and reported
for reference, but no checkpointing is implemented here). If you want the
beyond-accuracy metrics computed at the best-val epoch instead, you need to
add checkpoint save/restore -- this script does not do that.

Usage:
    python test_metrics.py --dataset imdb --epochs 20
    python test_semantic_metrics.py --dataset imdb --epochs 20 --seeds 42 123 7
    python test_semantic_metrics.py --dataset imdb --epochs 20 --arms ae_bpr semantic_ae semantic_ae_detach
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

from utils.dataset_utils import RankingTrainDataset, train_collate_fn, create_sparse_matrix
from utils.dataset_utils import loocv_collate_fn, build_interacted_by_user
from utils.leave_one_out_cv import get_loocv_fold_normalized

from model.cf_autoencoder import CFAutoEncoder
from model.semantic_ae import SemanticDisentangledAE


# ---------------------------------------------------------------------------
# Fixed backbone -- deliberately simple, not searched. Keep IDENTICAL across
# both arms so the comparison isolates the semantic pathway, not architecture.
# ---------------------------------------------------------------------------
LAYER_FRAC = 0.2
MIN_WIDTH, MAX_WIDTH = 32, 4096
DROPOUT = 0.2
LR = 1e-4
NL_TYPE = "relu"
TIED_WEIGHTS = True

# SemanticDisentangledAE default loss weights (see model/semantic_ae.py).
SEMANTIC_AE_DEFAULTS = dict(alpha=1.0, beta=0.1, detach_code_for_align=False)
NUM_ASPECTS = 20  # must match aspects/build_user_aspect_profiles.py --num_aspects

# semantic_ae_detach: null baseline where the alignment loss can't touch the
# CF encoder at all (detach_code_for_align=True) -- if beyond-accuracy metrics
# don't differ from semantic_ae, the effect isn't gradient-mediated. Not run
# by default; opt in via --arms.
ARMS = ("ae_bpr", "semantic_ae", "semantic_ae_detach")
DEFAULT_ARMS = ["ae_bpr", "semantic_ae"]


def set_seed(seed):
    """
    Seeds everything that introduces run-to-run variance in this harness:
    model weight init (nn.init.xavier_uniform_ etc., via torch's global
    RNG), DataLoader shuffling (shuffle=True uses an explicit per-seed
    generator -- see build_loaders), and numpy/python-level randomness in
    the data pipeline (negative sampling inside RankingTrainDataset uses its
    own local generator, seeded independently -- see build_loaders).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_layer_sizes(total_items, frac=LAYER_FRAC, min_w=MIN_WIDTH, max_w=MAX_WIDTH):
    w = int(np.clip(round(total_items * frac), min_w, max_w))
    return [total_items, w, max(w // 2, min_w)]


def load_fold(dataset_name, dataset_dir="datasets", embeddings_dir="embeddings_reviews",
              aspect_profiles_dir="aspect_profiles"):
    full_df = pd.read_csv(os.path.join(dataset_dir, f"{dataset_name}.csv"))
    full_df["userId"] = full_df["userId"].astype("int32")
    full_df["itemId"] = full_df["itemId"].astype("int32")

    # RankingTrainDataset requires a user_text frame positionally even for
    # arms that never read "user_text" from the batch (ae_bpr, semantic_ae) --
    # loaded here as harness plumbing, not because either arm consumes it.
    embedding = np.load(os.path.join(embeddings_dir, f"{dataset_name}.npy"))
    df_text = pd.DataFrame(embedding)
    df_text["userId"] = full_df["userId"].values
    df_text["itemId"] = full_df["itemId"].values

    total_items = int(full_df.itemId.max() + 1)
    total_users = int(full_df.userId.max() + 1)

    # SemanticDisentangledAE's per-user aspect-sentiment profiles (see
    # aspects/build_user_aspect_profiles.py). ae_bpr never reads these --
    # absent, any semantic_ae* arm fails fast in build_model with a message
    # pointing at the script that generates them.
    probs_path = os.path.join(aspect_profiles_dir, f"{dataset_name}_probs.npy")
    mask_path = os.path.join(aspect_profiles_dir, f"{dataset_name}_mask.npy")
    aspect_probs = aspect_mask = None
    num_aspects = None
    if os.path.exists(probs_path) and os.path.exists(mask_path):
        aspect_probs = np.load(probs_path)
        aspect_mask = np.load(mask_path)
        num_aspects = aspect_mask.shape[1]

    train, val, _ = get_loocv_fold_normalized(full_df, 0)
    train_matrix = create_sparse_matrix(train, total_users, total_items)

    # TRAIN-set-only interaction counts per item -- feeds both `novelty` and
    # the `serendipity` popularity baseline in CFAutoEncoder.evaluate().
    # Computed once per fold since it doesn't change across arms/epochs/seeds.
    item_popularity = torch.from_numpy(
        np.asarray((train_matrix != 0).sum(axis=0)).flatten()
    ).float()

    train_pairs = pd.MultiIndex.from_frame(train[["userId", "itemId"]])
    df_text_pairs = pd.MultiIndex.from_frame(df_text[["userId", "itemId"]])
    df_text_seen = df_text[df_text_pairs.isin(train_pairs)].copy()

    interacted_by_user = build_interacted_by_user(train)

    return dict(
        total_items=total_items, total_users=total_users,
        train=train, val=val, train_matrix=train_matrix, df_text_seen=df_text_seen,
        interacted_by_user=interacted_by_user,
        aspect_probs=aspect_probs, aspect_mask=aspect_mask, num_aspects=num_aspects,
        item_popularity=item_popularity,
    )


def build_model(arm, fold, device):
    layer_sizes = build_layer_sizes(fold["total_items"])
    common_kwargs = dict(
        nl_type=NL_TYPE, dp_drop_prob=DROPOUT, learn_rate=LR, tied_weights=TIED_WEIGHTS,
    )

    if arm == "ae_bpr":
        return CFAutoEncoder(layer_sizes=layer_sizes, **common_kwargs).to(device)

    if arm.startswith("semantic_ae"):
        if fold.get("aspect_probs") is None:
            raise RuntimeError(
                f"arm {arm!r} needs per-user aspect profiles, but none were found for this "
                f"dataset. Run: python aspects/build_user_aspect_profiles.py --dataset <name> "
                f"--num_aspects {NUM_ASPECTS}"
            )
        sem_kwargs = dict(SEMANTIC_AE_DEFAULTS)
        if arm == "semantic_ae_detach":
            sem_kwargs["detach_code_for_align"] = True
        elif arm != "semantic_ae":
            raise ValueError(f"Unknown semantic_ae arm: {arm!r}")

        return SemanticDisentangledAE(
            layer_sizes=layer_sizes,
            num_aspects=fold["num_aspects"],
            **sem_kwargs,
            **common_kwargs,
        ).to(device)

    raise ValueError(f"Unknown arm: {arm!r}, expected one of {ARMS}")


def build_loaders(fold, batch_size=2048, num_workers=0, seed=42):
    aspect_kwargs = dict(aspect_probs=fold.get("aspect_probs"), aspect_mask=fold.get("aspect_mask"))
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        RankingTrainDataset(fold["train_matrix"], fold["df_text_seen"], fold["train"], **aspect_kwargs, seed=seed),
        batch_size=batch_size, shuffle=True, collate_fn=train_collate_fn,
        num_workers=num_workers, generator=g,
    )
    val_loader = DataLoader(
        RankingTrainDataset(fold["train_matrix"], fold["df_text_seen"], fold["val"], **aspect_kwargs, seed=seed),
        batch_size=4096, shuffle=False,
        collate_fn=lambda b: loocv_collate_fn(b, fold["interacted_by_user"], fold["total_items"]),
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    return train_loader, val_loader


def run_epochs(model, train_loader, val_loader, epochs, k=10, eval_every=1):
    """
    Per-epoch loop: hit_rate/ndcg/mrr only. Beyond-accuracy metrics
    (novelty/diversity/serendipity) are deliberately NOT computed here --
    see module docstring. Call run_final_beyond_accuracy_eval() once, after
    this loop finishes, for those.
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
                for k_, v in comps.items():
                    comp_sums[k_] = comp_sums.get(k_, 0.0) + v * n

        avg_loss = total_loss / max(total_n, 1)
        avg_comps = {k_: v / max(total_n, 1) for k_, v in comp_sums.items()}

        row = {"epoch": epoch, "train_loss": avg_loss,
               **{f"loss_{k_}": v for k_, v in avg_comps.items()}}

        if epoch % eval_every == 0 or epoch == epochs:
            metrics = model.evaluate(val_loader, k=k)
            row.update(metrics)
            best_hit_rate = max(best_hit_rate, metrics["hit_rate"])

            extra = ""
            if "recon" in avg_comps:
                extra = (f" | loss_bpr {avg_comps['bpr']:.4f} | loss_recon {avg_comps['recon']:.4f}"
                         f" | loss_align {avg_comps.get('align', float('nan')):.4f}")
            print(f"  epoch {epoch:3d} | train_loss {avg_loss:.4f} | "
                  f"hit_rate {metrics['hit_rate']:.4f} | ndcg {metrics['ndcg']:.4f} | "
                  f"mrr {metrics['mrr']:.4f}{extra}")

        history.append(row)

    return best_hit_rate, pd.DataFrame(history)


def run_final_beyond_accuracy_eval(model, val_loader, item_popularity, k=10, k_pop=None):
    """
    One extra, more expensive evaluate() call on the model's final-epoch
    state: full-catalog scoring pass -> novelty/diversity/serendipity, on
    top of hit_rate/ndcg/mrr recomputed the same way as during training (for
    a sanity-check cross-reference against the last row of the per-epoch
    history, not because the number should differ).

    Item embeddings for `diversity` are no longer passed in here -- evaluate()
    pulls them itself via model.item_embeddings_ (see CFAutoEncoder), since
    that's the same source for every arm that shares CFAutoEncoder.evaluate().
    """
    return model.evaluate(val_loader, k=k, item_popularity=item_popularity, k_pop=k_pop)


def run_one(arm, fold, device, epochs, batch_size, num_workers, seed, k=10, k_pop=None):
    print(f"\n{'=' * 70}\n{arm}  (seed={seed})\n{'=' * 70}")

    model = build_model(arm, fold, device)
    set_seed(seed)
    train_loader, val_loader = build_loaders(fold, batch_size=batch_size, num_workers=num_workers, seed=seed)

    best_hit_rate, history_df = run_epochs(model, train_loader, val_loader, epochs, k=k)
    final_metrics = {k_: history_df.iloc[-1][k_] for k_ in ("hit_rate", "ndcg", "mrr")}
    print(f"[{arm}] best_val_hit_rate over training: {best_hit_rate:.4f}")
    print(f"[{arm}] final epoch metrics: {final_metrics}")

    beyond_acc = {}
    if fold.get("item_popularity") is not None:
        beyond_acc = run_final_beyond_accuracy_eval(
            model, val_loader, fold["item_popularity"].to(device), k=k, k_pop=k_pop,
        )
        print(f"[{arm}] beyond-accuracy (final-epoch state): "
              f"novelty {beyond_acc.get('novelty', float('nan')):.4f} | "
              f"diversity {beyond_acc.get('diversity', float('nan')):.4f} | "
              f"serendipity {beyond_acc.get('serendipity', float('nan')):.4f}")

    result = {"arm": arm, "seed": seed, "best_val_hit_rate": best_hit_rate,
              **final_metrics, **beyond_acc}

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result, history_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Dataset name (matches datasets/{name}.csv)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--k", type=int, default=10, help="Top-k cutoff for all metrics.")
    parser.add_argument("--k_pop", type=int, default=None,
                         help="Size of the popularity baseline used by serendipity. "
                              "Defaults to --k if not given.")
    parser.add_argument("--arms", type=str, nargs="+", default=DEFAULT_ARMS, choices=ARMS,
                         help="Subset of arms to run, e.g. --arms ae_bpr semantic_ae")
    parser.add_argument("--seeds", type=int, nargs="+", default=[23, 43, 111, 213, 159, 787, 651, 369],
                         help="One or more seeds. With only n<7 seeds, Wilcoxon's minimum "
                              "achievable p-value is 2*0.5**n -- a perfect sweep may still "
                              "not reach p<0.05. 7+ seeds recommended.")
    args = parser.parse_args()

    print("This run compares AE_BPR (pure collaborative backbone) against "
          "SemanticDisentangledAE (backbone + sentiment-alignment pathway). "
          "hit_rate/ndcg/mrr are computed every epoch; novelty/diversity/"
          "serendipity are computed ONCE, on the final-epoch model state, "
          "after training finishes (see module docstring).\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading fold for {args.dataset} ...")
    fold = load_fold(args.dataset)

    multi_seed = len(args.seeds) > 1
    all_results = []

    for arm in args.arms:
        for seed in args.seeds:
            set_seed(seed)
            res, history_df = run_one(arm, fold, device, epochs=args.epochs,
                                       batch_size=args.batch_size, num_workers=args.num_workers,
                                       seed=seed, k=args.k, k_pop=args.k_pop)
            all_results.append(res)

            suffix = f"_seed{seed}" if multi_seed else ""
            history_df.to_csv(f"{args.dataset}_{arm}{suffix}_epoch_history.csv", index=False)
            print(f"Saved {arm} (seed={seed}) per-epoch history to "
                  f"{args.dataset}_{arm}{suffix}_epoch_history.csv")

    df = pd.DataFrame(all_results)
    print(f"\n\n=== Arm comparison ({args.epochs} epochs, seeds={args.seeds}) ===")
    print(df.to_string(index=False))

    out_path = f"{args.dataset}_semantic_arm_comparison.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved comparison to {out_path}")

    if multi_seed:
        metric_cols = [c for c in
                       ["best_val_hit_rate", "hit_rate", "ndcg", "mrr", "novelty", "diversity", "serendipity"]
                       if c in df.columns]
        summary = df.groupby("arm")[metric_cols].agg(["mean", "std"])
        print(f"\n=== Mean/std across {len(args.seeds)} seeds ===")
        print(summary.to_string())
        summary_path = f"{args.dataset}_semantic_arm_comparison_summary.csv"
        summary.to_csv(summary_path)
        print(f"\nSaved seed summary to {summary_path}")

        # ---- paired per-seed comparison ----
        # Pairing by seed cancels out "this seed is just hard for every arm"
        # variance, which is what actually answers "does arm A beat arm B".
        pairs = [
            ("semantic_ae", "ae_bpr"),
            ("semantic_ae_detach", "semantic_ae"),
            ("semantic_ae_detach", "ae_bpr"),
        ]
        available_pairs = [(a, b) for a, b in pairs if a in df["arm"].values and b in df["arm"].values]

        if available_pairs:
            try:
                from scipy.stats import wilcoxon
                have_scipy = True
            except ImportError:
                have_scipy = False
                print("\n(scipy not available -- skipping Wilcoxon p-values, "
                      "win-counts/mean-diffs still reported below)")

            for metric in metric_cols:
                piv = df.pivot(index="seed", columns="arm", values=metric)
                print(f"\n=== Paired per-seed comparison ({metric}) ===")
                paired_rows = []
                for a, b in available_pairs:
                    if a not in piv.columns or b not in piv.columns:
                        continue
                    diff = piv[a] - piv[b]
                    wins = int((diff > 0).sum())
                    n = diff.notna().sum()
                    p = None
                    if have_scipy:
                        try:
                            _, p = wilcoxon(diff.dropna())
                        except ValueError:
                            p = None  # e.g. all-zero diffs
                    paired_rows.append({
                        "metric": metric, "a": a, "b": b, "a_wins": wins, "n_seeds": n,
                        "mean_diff": diff.mean(), "wilcoxon_p": p,
                    })
                    p_str = f"{p:.4f}" if p is not None else "n/a"
                    print(f"  {a} - {b}: wins {wins}/{n}  mean_diff {diff.mean():+.5f}  "
                          f"wilcoxon_p {p_str}")
                if paired_rows:
                    paired_df = pd.DataFrame(paired_rows)
                    paired_path = f"{args.dataset}_semantic_arm_comparison_paired_{metric}.csv"
                    paired_df.to_csv(paired_path, index=False)

            if len(args.seeds) < 7:
                print(f"\nNote: with n_seeds={len(args.seeds)}, Wilcoxon's minimum achievable "
                      f"p-value is {2 * (0.5 ** len(args.seeds)):.4f} -- a perfect sweep may "
                      f"still not reach p<0.05. 7+ seeds recommended.")