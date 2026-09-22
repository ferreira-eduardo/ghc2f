"""
Comparison harness: AE_BPR (CFAutoEncoder) vs. the three GatedHybridCFAutoEncoder
fusion modes -- convex (original: h = g*h + (1-g)*t, text and collaborative
signal directly trade off), additive (h = h + g*t, collaborative signal
never diminished, text only adds), and film (h = h*(1+gamma)+beta, text
modulates rather than blends). Isolates the fusion-mechanism question
before layering the contrastive extension (GHC2F) back on top.

All arms trained purely under BPR-family objectives -- MMSE has been
dropped entirely (see cf_autoencoder.py / ghc2f.py docstrings for the
loss-scale-imbalance finding that motivated this).

Trains all arms with a hand-rolled epoch loop (not utils.train_model) so
every arm goes through IDENTICAL step/eval logic regardless of model class
-- this matters because CFAutoEncoder, GatedHybridCFAutoEncoder, and GHC2F
all expose calculate_loss(batch) -> (loss, batch_size) and evaluate(loader)
-> {hit_rate, ndcg, mrr}, but this test doesn't assume anything about
train_model's internals staying comparable across model classes.

--seeds accepts one or more seeds so each arm can be run multiple times to
check whether a result is stable or fold/init noise. With a single seed,
behavior/filenames are unchanged; with multiple seeds, per-seed CSVs get a
_seedN suffix and mean/std + paired Wilcoxon summaries are added.

Usage:
    python test_ghc2f.py --dataset imdb --epochs 20
    python test_ghc2f.py --dataset imdb --epochs 20 --seeds 42 123 7
    python test_ghc2f.py --dataset imdb --epochs 20 --arms ae_bpr ghc2f_decoder
"""
import argparse
import inspect
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
from model.ae_constrastive import AE_Contrastive
from model.semantic_ae import SemanticDisentangledAE


# ---------------------------------------------------------------------------
# Fixed backbone -- deliberately simple, not searched. Keep IDENTICAL across
# all arms so the comparison isolates model/objective, not architecture.
# ---------------------------------------------------------------------------
LAYER_FRAC = 0.2
MIN_WIDTH, MAX_WIDTH = 32, 4096
DROPOUT = 0.2
LR = 1e-4
NL_TYPE = "relu"
TIED_WEIGHTS = True

GHC2F_LOSS_DEFAULTS = dict(
    contrastive_target="text",
    stop_grad_cf=True,
    cl_weight=0.1,
    reg_weight=1e-5,
    gate_entropy_weight=0.0,
)

FUSION_MODES = ("convex", "additive", "film")

# SemanticDisentangledAE default loss weights (see model/semantic_ae.py).
SEMANTIC_AE_DEFAULTS = dict(alpha=1.0, beta=0.1, detach_code_for_align=True)
NUM_ASPECTS = 20  # must match aspects/build_user_aspect_profiles.py --num_aspects

# SemanticDisentangledAE ablations, addressing the reviewer-risk points this
# model was flagged for (see Phase 3 discussion): tied vs. untied decoder
# weights (never previously searched for the prior model either), a null
# baseline where the alignment loss can't touch the CF encoder at all
# (detach_code_for_align=True -- if hit_rate doesn't differ from the real
# run, the effect isn't coming from semantic regularization), and alpha=0 /
# beta=0 single-term ablations to isolate L_Recon and L_Align individually.
SEMANTIC_ARMS = (
    "semantic_ae",            # alpha=1.0, beta=0.1, tied_weights (=TIED_WEIGHTS), detach=False
    "semantic_ae_untied",     # tied_weights=False ablation
    "semantic_ae_detach",     # null baseline: detach_code_for_align=True
    "semantic_ae_alpha0",     # alpha=0 -- drop L_Recon
    "semantic_ae_beta0",      # beta=0  -- drop L_Align
)

# Default comparison: AE_BPR against the three fusion mechanisms directly,
# isolating the fusion question before the contrastive extension re-enters
# the picture. ghc2f_decoder/ghc2f_no_cl/semantic_ae_* remain available via --arms.
ARMS = (
    ["ae_bpr"]
    + [f"gated_hybrid_{mode}" for mode in FUSION_MODES]
    + ["ghc2f_decoder", "ghc2f_no_cl"] + ['ae_contrastive']
    + list(SEMANTIC_ARMS)
)
DEFAULT_ARMS = ["ae_bpr"] + [f"gated_hybrid_{mode}" for mode in FUSION_MODES]


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


def load_fold(dataset_name, dataset_dir="datasets", embeddings_dir="embeddings_reviews",
              aspect_profiles_dir="aspect_profiles"):
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

    # Optional: SemanticDisentangledAE's per-user aspect-sentiment profiles
    # (see aspects/build_user_aspect_profiles.py). None of the other arms
    # read these -- absent, any semantic_ae_* arm fails fast in build_model
    # with a message pointing at the script that generates them.
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
    # the `serendipity` popularity baseline in CFAutoEncoder.evaluate() (see
    # model/cf_autoencoder.py). Computed once per fold since it doesn't
    # change across arms/epochs/seeds.
    item_popularity = torch.from_numpy(
        np.asarray((train_matrix != 0).sum(axis=0)).flatten()
    ).float()

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
        aspect_probs=aspect_probs, aspect_mask=aspect_mask, num_aspects=num_aspects,
        item_popularity=item_popularity,
    )


def _set_item_corpus(model, fold, device):
    """
    Shared by gated_hybrid_* and both ghc2f_* arms: gives the model the raw
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

    if arm == "ae_bpr":
        return CFAutoEncoder(layer_sizes=layer_sizes, **common_kwargs).to(device)

    if arm.startswith("gated_hybrid_"):
        fusion_mode = arm[len("gated_hybrid_"):]
        assert fusion_mode in FUSION_MODES, \
            f"Unknown fusion mode {fusion_mode!r} in arm {arm!r}, expected one of {FUSION_MODES}"
        model = GatedHybridCFAutoEncoder(
            layer_sizes=layer_sizes,
            num_users=fold["total_users"],
            num_items=fold["total_items"],
            text_dim=fold["text_dim"],
            text_latent_dim=64,
            fusion_mode=fusion_mode,
            **common_kwargs,
        ).to(device)
        _set_item_corpus(model, fold, device)
        return model

    if arm == "ae_contrastive":
        model = AE_Contrastive(
            layer_sizes=layer_sizes,
            num_users=fold["total_users"],
            num_items=fold["total_items"],
            text_dim=fold["text_dim"],
            text_latent_dim=64,
            **common_kwargs,
        ).to(device)
        _set_item_corpus(model, fold, device)
        return model

    if arm.startswith("semantic_ae"):
        if fold.get("aspect_probs") is None:
            raise RuntimeError(
                f"arm {arm!r} needs per-user aspect profiles, but none were found for this "
                f"dataset. Run: python aspects/build_user_aspect_profiles.py --dataset <name> "
                f"--num_aspects {NUM_ASPECTS}"
            )
        sem_kwargs = dict(SEMANTIC_AE_DEFAULTS)
        arm_common_kwargs = dict(common_kwargs)
        if arm == "semantic_ae":
            sem_kwargs["detach_code_for_align"] = True
        elif arm == "semantic_ae_untied":
            arm_common_kwargs["tied_weights"] = False
        elif arm == "semantic_ae_detach":
            sem_kwargs["detach_code_for_align"] = True
        elif arm == "semantic_ae_alpha0":
            sem_kwargs["alpha"] = 0.0
        elif arm == "semantic_ae_beta0":
            sem_kwargs["beta"] = 0.0
        elif arm != "semantic_ae":
            raise ValueError(f"Unknown semantic_ae arm: {arm!r}")

        return SemanticDisentangledAE(
            layer_sizes=layer_sizes,
            num_aspects=fold["num_aspects"],
            **sem_kwargs,
            **arm_common_kwargs,
        ).to(device)


    loss_kwargs = dict(GHC2F_LOSS_DEFAULTS)
    if arm == "ghc2f_no_cl":
        loss_kwargs["cl_weight"] = 0.0

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


def build_loaders(fold, batch_size=2048, num_workers=0, seed= 42):
    aspect_kwargs = dict(aspect_probs=fold.get("aspect_probs"), aspect_mask=fold.get("aspect_mask"))
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        RankingTrainDataset(fold["train_matrix"], fold["df_text_seen"], fold["train"], **aspect_kwargs, seed=seed),
        batch_size=batch_size, shuffle=True, collate_fn=train_collate_fn,
        num_workers=num_workers ,generator=g
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


def run_epochs(model, train_loader, val_loader, epochs, eval_every=1, item_popularity=None):
    history = []
    best_hit_rate = -1.0

    gate_batch = None
    if hasattr(model, "gate_diagnostics"):
        gate_batch = next(iter(val_loader))

    # novelty/diversity/serendipity are only accepted by CFAutoEncoder.evaluate()
    # -- GHC2F defines its own evaluate(test_loader, k=10) override (see
    # model/ghc2f.py) that doesn't take these kwargs, so passing them there
    # would raise TypeError. Check the live signature rather than hardcoding
    # a class list, so any future evaluate() override is handled correctly
    # too.
    supports_beyond_accuracy = "item_popularity" in inspect.signature(model.evaluate).parameters

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
            eval_kwargs = {}
            if supports_beyond_accuracy and item_popularity is not None:
                eval_kwargs = dict(item_popularity=item_popularity,
                                    item_embeddings=model.get_item_embeddings().detach())
            metrics = model.evaluate(val_loader, **eval_kwargs)
            row.update(metrics)
            best_hit_rate = max(best_hit_rate, metrics["hit_rate"])

            gate_extra = ""
            if gate_batch is not None:
                gate_df = pd.DataFrame(model.gate_diagnostics(gate_batch))
                row["gate_mean_avg"] = gate_df["mean"].mean()
                row["gate_std_avg"] = gate_df["std"].mean()
                row["gate_frac_above_0.9_avg"] = gate_df["frac_above_0.9"].mean()
                row["gate_frac_below_0.1_avg"] = gate_df["frac_below_0.1"].mean()
                gate_extra = f" | gate_mean {row['gate_mean_avg']:.4f}"

            extra = ""
            if "cl" in avg_comps:
                extra = (f" | loss_bpr {avg_comps['bpr']:.4f} | loss_cl {avg_comps['cl']:.4f}"
                         f" | loss_gate {avg_comps.get('gate', float('nan')):.4f}"
                         f" | loss_reg {avg_comps.get('reg', float('nan')):.4f}")
            elif "recon" in avg_comps:
                extra = (f" | loss_bpr {avg_comps['bpr']:.4f} | loss_recon {avg_comps['recon']:.4f}"
                         f" | loss_align {avg_comps.get('align', float('nan')):.4f}")
            print(f"  epoch {epoch:3d} | train_loss {avg_loss:.4f} | "
                  f"hit_rate {metrics['hit_rate']:.4f} | ndcg {metrics['ndcg']:.4f} | "
                  f"mrr {metrics['mrr']:.4f}{extra}{gate_extra}")

        history.append(row)

    return best_hit_rate, pd.DataFrame(history)


def run_one(arm, fold, device, epochs, batch_size, num_workers, seed):
    print(f"\n{'=' * 70}\n{arm}  (seed={seed})\n{'=' * 70}")

    model = build_model(arm, fold, device)
    set_seed(seed)
    train_loader, val_loader = build_loaders(fold, batch_size=batch_size, num_workers=num_workers, seed=seed)

    best_hit_rate, history_df = run_epochs(model, train_loader, val_loader, epochs,
                                            item_popularity=fold.get("item_popularity"))
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
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--arms", type=str, nargs="+", default=DEFAULT_ARMS, choices=ARMS,
                         help="Subset of arms to run, e.g. --arms ae_bpr gated_hybrid_additive")
    parser.add_argument("--seeds", type=int, nargs="+", default=[23, 43, 111, 213, 159, 787, 651, 369],
                         help="One or more seeds, e.g. --seeds 42 123 7. Each arm is run "
                              "once per seed; with >1 seed a mean/std summary is added. "
                              "With only 4 seeds, Wilcoxon's minimum achievable p-value is "
                              "0.125, so even a perfect 4/4 sweep can never reach "
                              "significance -- 7 seeds is the recommended minimum.")
    args = parser.parse_args()

    print("NOTE: MMSE has been dropped entirely from every arm (see "
          "cf_autoencoder.py / ghc2f.py docstrings). This run compares "
          "AE_BPR against the three GatedHybridCFAutoEncoder fusion modes:\n"
          "  ae_bpr                -> CFAutoEncoder, pure pairwise BPR (Eq. 2).\n"
          "  gated_hybrid_convex   -> h = g*h + (1-g)*t -- ORIGINAL behavior. "
          "Text and collaborative signal directly TRADE OFF (zero-sum by "
          "construction, not just in practice).\n"
          "  gated_hybrid_additive -> h = h + g*t -- collaborative signal "
          "is NEVER diminished; text can only ADD, never subtract.\n"
          "  gated_hybrid_film     -> h = h*(1+gamma) + beta -- text "
          "CONDITIONS the collaborative representation rather than being "
          "blended with it. Most expressive of the three.\n"
          "  ghc2f_decoder / ghc2f_no_cl -> available via --arms, not run "
          "by default here (fusion_mode currently fixed to 'convex' for "
          "these; the fusion-mode question is isolated on gated_hybrid_* "
          "first).\n")

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
            history_df.to_csv(f"{args.dataset}_{arm}{suffix}_epoch_history_film.csv", index=False)
            print(f"Saved {arm} (seed={seed}) per-epoch history_film to "
                  f"{args.dataset}_{arm}{suffix}_epoch_history_film.csv")
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

        # ---- paired per-seed comparison ----
        # Unpaired mean/std (above) can hide a real, consistent effect when
        # between-seed variance is larger than the gap between arms (some
        # seeds are just harder for every arm). Pairing by seed cancels
        # that shared "this seed is hard" effect and is what actually
        # answers "does arm A beat arm B", not just "do their means differ".
        pairs = [
            ("gated_hybrid_convex", "ae_bpr"),
            ("gated_hybrid_additive", "ae_bpr"),
            ("gated_hybrid_film", "ae_bpr"),
            ("gated_hybrid_additive", "gated_hybrid_convex"),
            ("gated_hybrid_film", "gated_hybrid_convex"),
            ("gated_hybrid_additive", "gated_hybrid_film"),
            ("ghc2f_decoder", "ae_bpr"),
            ("ghc2f_decoder", "gated_hybrid_convex"),
            ("ghc2f_decoder", "ghc2f_no_cl"),
            ("ghc2f_no_cl", "gated_hybrid_convex"),
            ("semantic_ae", "ae_bpr"),
            ("semantic_ae_untied", "semantic_ae"),
            ("semantic_ae_detach", "semantic_ae"),
            ("semantic_ae_alpha0", "semantic_ae"),
            ("semantic_ae_beta0", "semantic_ae"),
        ]
        piv = df.pivot(index="seed", columns="arm", values="best_val_hit_rate")
        available_pairs = [(a, b) for a, b in pairs if a in piv.columns and b in piv.columns]

        if available_pairs:
            try:
                from scipy.stats import wilcoxon
                have_scipy = True
            except ImportError:
                have_scipy = False
                print("\n(scipy not available -- skipping Wilcoxon p-values, "
                      "win-counts/mean-diffs still reported below)")

            print(f"\n=== Paired per-seed comparison (best_val_hit_rate) ===")
            paired_rows = []
            for a, b in available_pairs:
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
                    "a": a, "b": b, "a_wins": wins, "n_seeds": n,
                    "mean_diff": diff.mean(), "wilcoxon_p": p,
                })
                p_str = f"{p:.4f}" if p is not None else "n/a"
                print(f"  {a} - {b}: wins {wins}/{n}  mean_diff {diff.mean():+.4f}  "
                      f"wilcoxon_p {p_str}")
            paired_df = pd.DataFrame(paired_rows)
            paired_path = f"{args.dataset}_arm_comparison_paired.csv"
            paired_df.to_csv(paired_path, index=False)
            print(f"\nSaved paired comparison to {paired_path}")
            if len(args.seeds) < 7:
                print(f"Note: with n_seeds={len(args.seeds)}, Wilcoxon's minimum achievable "
                      f"p-value is {2 * (0.5 ** len(args.seeds)):.4f} -- a perfect sweep may "
                      f"still not reach p<0.05. 7+ seeds recommended.")

    for name, gdf in gate_frames.items():
        gate_path = f"{args.dataset}_{name}_gate.csv"
        gdf.to_csv(gate_path, index=False)
        print(f"Saved {name} gate diagnostics to {gate_path}")