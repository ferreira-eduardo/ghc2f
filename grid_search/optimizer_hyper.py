import argparse
import gc
import os
import sys

import numpy as np
import optuna
import pandas as pd
import torch
from torch.utils.data import DataLoader

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils.utils import prepare_inputs, EarlyStoppingRanking
from utils.dataset_utils import RankingTrainDataset, train_collate_fn, create_sparse_matrix
from utils.dataset_utils import loocv_collate_fn, build_interacted_by_user

from model.ghc2f import GHC2F
from model.cf_autoencoder import CFAutoEncoder
from model.gated_hybrid_ae import GatedHybridCFAutoEncoder
from utils.leave_one_out_cv import get_loocv_fold_normalized
from utils.train_model import train_model


MODEL_REGISTRY = {
    "cf_autoencoder": CFAutoEncoder,
    "gated_hybrid": GatedHybridCFAutoEncoder,
    "ghc2f": GHC2F,
}

# ---------------------------------------------------------------------------
# Layer search space
# ---------------------------------------------------------------------------
_WIDTH_FRACS = {
    "mf_like": 0.02,
    "narrow": 0.08,
    "mid": 0.20,
    "wide": 0.40,
}
_MIN_WIDTH, _MAX_WIDTH = 32, 4096


def _width(n: int, frac: float) -> int:
    return int(np.clip(round(n * frac), _MIN_WIDTH, _MAX_WIDTH))


def build_layer_options(n: int) -> dict:
    """Cartesian grid over {width tier} x {depth}, all widths relative to n
    and monotonically non-increasing within a config. Every config is a
    genuine bottleneck (first hidden width <= n), so the same tier name
    (e.g. 'mid') means a comparable compression ratio across datasets of
    very different item counts, instead of a fixed absolute width like 4096
    meaning "compression" for one dataset and "expansion" for another.
    """
    options = {}
    for tier, frac in _WIDTH_FRACS.items():
        w = _width(n, frac)
        options[f"{tier}_d1"] = lambda n_, w=w: [n_, w]
        options[f"{tier}_d2"] = lambda n_, w=w: [n_, w, max(w // 2, _MIN_WIDTH)]
        options[f"{tier}_d3"] = lambda n_, w=w: [
            n_, w, max(w // 2, _MIN_WIDTH), max(w // 4, _MIN_WIDTH)
        ]
    return options


# Defaults used for GHC2F-specific loss hyperparameters during Stage 1
# (backbone search), so architecture/optimizer choices aren't confounded by
# an untuned loss-weight configuration. Keys must match GHC2F.__init__'s
# actual kwarg names (model/ghc2f.py) — topic_gamma is a GatedHybridCFAutoEncoder
# kwarg forwarded through **ae_kwargs and is left at its own default here.

STAGE1_GHC2F_DEFAULTS = dict(
    contrastive_target="text",
    stop_grad_cf=True,
    dual_head_decoder=True,
    cl_weight=0.1,
    reg_weight=1e-5,
    mmse_weight=1.0,
    mmse_weight_cf=1.0,
    gate_entropy_weight=0.0,
)

STAGE1_EPOCHS = 15
STAGE3_EPOCHS = 40  # full-data confirmation gets a real training budget, not the 15-epoch search budget
STAGE3_TOP_K = 3    # re-confirm top-K stage-1 trials on 100% data, not just the single "best"

# sample_frac for the search stages. These used to be small (0.2/0.3) purely
# to keep Optuna trials cheap — but RankingTrainDataset previously iterated
# every global user id regardless of how many actually had a real row for
# this split (see utils/dataset_utils.py), so on sparse datasets the vast
# majority of every "epoch" was wasted on ghost users with an all-zero input,
# and the LOOCV-eligible population (>=3 interactions) was already a small
# fraction of TOTAL_USERS to begin with. Now that the loader only visits real
# rows, per-epoch cost tracks genuine population size, not TOTAL_USERS, so
# it's cheap to raise these and get a materially larger/less noisy
# validation signal for the search stages instead of the ~199 real users
# 0.2 produced on All_Beauty_reviews.
STAGE1_SAMPLE_FRAC = 0.5


class GatedAEOptimizer:
    def __init__(self, dataset_name, device, sample_frac=STAGE1_SAMPLE_FRAC, random_state=42,
                 dataset_dir="datasets", embeddings_dir="embeddings_reviews",
                 batch_size=2048, num_workers=0):
        self.dataset_name = dataset_name
        self.device = device
        self.random_state = random_state
        self.batch_size = batch_size
        self.num_workers = num_workers

        print(f"Loading {dataset_name} (samples of {sample_frac:.0%} users)...")
        self.full_df = pd.read_csv(os.path.join(dataset_dir, f"{dataset_name}.csv"))
        self.full_df["userId"] = self.full_df["userId"].astype("int32")
        self.full_df["itemId"] = self.full_df["itemId"].astype("int32")

        embedding = np.load(os.path.join(embeddings_dir, f"{dataset_name}.npy"))
        self.df_text = pd.DataFrame(embedding)
        self.df_text["userId"] = self.full_df["userId"].values
        self.df_text["itemId"] = self.full_df["itemId"].values
        self.text_cols = [col for col in self.df_text.columns if str(col).isdigit()]
        self.text_dim = len(self.text_cols)

        self.TOTAL_ITEMS = int(self.full_df.itemId.max() + 1)
        self.TOTAL_USERS = int(self.full_df.userId.max() + 1)

        self.layer_options = build_layer_options(self.TOTAL_ITEMS)

        self._fold_cache = None
        self._resample(sample_frac)

    def _resample(self, frac: float):
        unique_users = self.full_df['userId'].unique()
        sampled_users = pd.Series(unique_users).sample(frac=frac, random_state=self.random_state)
        self.df = self.full_df[self.full_df['userId'].isin(sampled_users)].copy()
        # self.df changed -> everything derived from the fold split is stale.
        self._fold_cache = None

    def _get_fold_cache(self):
        """Everything derived from self.df's LOOCV fold split is identical
        across every trial within a stage (only _resample() changes self.df),
        so compute it once per stage instead of redoing sort/groupby/apply
        work — including the item_profiler warmup inputs — on every one of
        the 60-113 trials that stage runs.
        """
        if self._fold_cache is not None:
            return self._fold_cache

        train, val, _ = get_loocv_fold_normalized(self.df, 0)
        train_matrix = create_sparse_matrix(train, self.TOTAL_USERS, self.TOTAL_ITEMS)

        train_pairs = pd.MultiIndex.from_frame(train[["userId", "itemId"]])
        df_text_pairs = pd.MultiIndex.from_frame(self.df_text[["userId", "itemId"]])
        df_text_seen = self.df_text[df_text_pairs.isin(train_pairs)].copy()

        interacted_by_user = build_interacted_by_user(train)

        df_text_train = self.df_text[self.df_text["itemId"].isin(train["itemId"].unique())].copy()
        item_ids, item_text, item_mask = prepare_inputs(df_text_train, "itemId", self.text_cols)

        self._fold_cache = dict(
            train=train, val=val, train_matrix=train_matrix,
            df_text_seen=df_text_seen, interacted_by_user=interacted_by_user,
            item_ids=item_ids, item_text=item_text, item_mask=item_mask,
        )
        return self._fold_cache

    def _build_model_and_loaders(self, model_cls, layer_sizes, nl_type, dropout, lr,
                                  tied_weights, extra_kwargs):
        """Shared setup: fold split, model construction, item profiles, dataloaders.

        tied_weights controls whether the decoder reuses the encoder's
        transposed weights (W_dec = W_enc^T), per Section 3.1 of the paper.
        This was previously never searched or reported (reviewer point 3d);
        it's now a first-class hyperparameter of Stage 1.
        """
        cache = self._get_fold_cache()
        train, val = cache["train"], cache["val"]
        train_matrix, df_text_seen = cache["train_matrix"], cache["df_text_seen"]
        interacted_by_user = cache["interacted_by_user"]

        common_kwargs = dict(nl_type=nl_type, dp_drop_prob=dropout, learn_rate=lr,
                              tied_weights=tied_weights)
        uses_text = issubclass(model_cls, GatedHybridCFAutoEncoder)

        if uses_text:
            model = model_cls(
                layer_sizes=layer_sizes,
                num_users=self.TOTAL_USERS,
                num_items=self.TOTAL_ITEMS,
                text_dim=self.text_dim,
                text_latent_dim=64,
                **common_kwargs,
                **extra_kwargs,
            ).to(self.device)

            item_ids, item_text, item_mask = cache["item_ids"], cache["item_text"], cache["item_mask"]
            with torch.no_grad():
                profiles = model.item_profiler(
                    item_ids.to(self.device), item_text.to(self.device), item_mask.to(self.device)
                ).detach()

            full_i_global = torch.zeros((self.TOTAL_ITEMS, self.text_dim), device=self.device)
            full_i_global[item_ids] = profiles
            model.item_global_profiles = full_i_global
        else:
            model = model_cls(layer_sizes=layer_sizes, **common_kwargs).to(self.device)

        persistent = self.num_workers > 0
        train_loader = DataLoader(
            RankingTrainDataset(train_matrix, df_text_seen, train),
            batch_size=self.batch_size, shuffle=True, collate_fn=train_collate_fn,
            num_workers=self.num_workers, persistent_workers=persistent,
        )

        val_loader = DataLoader(
            RankingTrainDataset(train_matrix, df_text_seen, val),
            batch_size=self.batch_size, shuffle=False,
            collate_fn=lambda b: loocv_collate_fn(b, interacted_by_user, self.TOTAL_ITEMS),
            num_workers=self.num_workers, persistent_workers=persistent,
        )

        return model, train_loader, val_loader

    def _run_and_cleanup(self, model, train_loader, val_loader, num_epochs=STAGE1_EPOCHS, trial=None):
        """Trains the model. When `trial` is given, reports the running best
        validation hit rate back to Optuna after every epoch so a pruner
        (SuccessiveHalving/Hyperband) can kill clearly-bad trials early —
        this is what lets a bigger grid (more architectures / more loss-
        weight combos) fit in the same wall-clock budget as the old 20/30
        fixed-trial runs.

        Falls back to a single full-length run if train_model doesn't accept
        an epoch_callback (keeps this script usable without modifying
        utils/train_model.py, at the cost of no pruning in that case).
        """
        early_stopping = EarlyStoppingRanking(patience=5, verbose=False)

        if trial is not None:
            state = {"best": -1.0}

            def _epoch_callback(epoch, val_hit_rate, **_):
                state["best"] = max(state["best"], val_hit_rate)
                trial.report(state["best"], step=epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            try:
                best_val_hit_rate, _ = train_model(
                    model, num_epochs, train_loader, val_loader, early_stopping,
                    epoch_callback=_epoch_callback,
                )
            except TypeError:
                # train_model doesn't support epoch_callback yet — run without pruning.
                best_val_hit_rate, _ = train_model(model, num_epochs, train_loader, val_loader, early_stopping)
        else:
            best_val_hit_rate, _ = train_model(model, num_epochs, train_loader, val_loader, early_stopping)

        del model, early_stopping
        torch.cuda.empty_cache()
        gc.collect()

        return best_val_hit_rate

    # ------------------------------------------------------------------
    # Stage 1: backbone / optimizer hyperparameters (all three models)
    # ------------------------------------------------------------------
    def objective_backbone(self, trial, model_cls):
        dropout = trial.suggest_categorical("dropout", [0.2, 0.5])
        lr = trial.suggest_categorical("lr", [1e-4, 1e-3])
        nl_type = trial.suggest_categorical("nl_type", ["selu", "relu"])
        layer_option = trial.suggest_categorical("layers", list(self.layer_options.keys()))
        tied_weights = trial.suggest_categorical("tied_weights", [True, False])

        layer_sizes = self.layer_options[layer_option](self.TOTAL_ITEMS)
        extra_kwargs = STAGE1_GHC2F_DEFAULTS if model_cls is GHC2F else {}

        model, train_loader, val_loader = self._build_model_and_loaders(
            model_cls, layer_sizes, nl_type, dropout, lr, tied_weights, extra_kwargs
        )
        return self._run_and_cleanup(model, train_loader, val_loader, trial=trial)

    def run_backbone_study(self, model_cls, n_trials=60):
        # ASHA-style pruning: with reporting per epoch (see _run_and_cleanup),
        # this lets n_trials be raised substantially (20 -> 60 here) for the
        # same wall-clock budget as before, since bad configs are killed
        # around epoch 3-5 instead of running the full 15 epochs.
        pruner = optuna.pruners.SuccessiveHalvingPruner(min_resource=3, reduction_factor=3)
        study = optuna.create_study(direction="maximize", pruner=pruner)
        study.optimize(lambda trial: self.objective_backbone(trial, model_cls), n_trials=n_trials)
        print("\n[Stage 1] Best backbone hyperparameters:")
        print(study.best_params)
        return study, study.best_params

    # ------------------------------------------------------------------
    # Stage 2: GHC2F-specific loss-weight hyperparameters (GHC2F only —
    # backbone fixed to the Stage 1 winner)
    # ------------------------------------------------------------------
    def objective_loss_weights(self, trial, fixed_backbone):
        topic_gamma = trial.suggest_float("topic_gamma", 0.1, 0.9)

        contrastive_target = trial.suggest_categorical(
            "contrastive_target", ["text", "collaborative"]
        )
        stop_grad_cf = trial.suggest_categorical("stop_grad_cf", [True, False])
        dual_head_decoder = trial.suggest_categorical("dual_head_decoder", [True, False])

        cl_weight = trial.suggest_float("cl_weight", 1e-3, 1.0, log=True)
        reg_weight = trial.suggest_float("reg_weight", 1e-6, 1e-3, log=True)

        mmse_weight = trial.suggest_float("mmse_weight", 0.1, 2.0)
        mmse_weight_cf = trial.suggest_float("mmse_weight_cf", 0.0, 2.0)

        gate_entropy_weight = trial.suggest_float("gate_entropy_weight", 0.0, 0.1)

        ghc2f_kwargs = dict(
            topic_gamma=topic_gamma,
            contrastive_target=contrastive_target,
            stop_grad_cf=stop_grad_cf,
            dual_head_decoder=dual_head_decoder,
            cl_weight=cl_weight,
            reg_weight=reg_weight,
            mmse_weight=mmse_weight,
            mmse_weight_cf=mmse_weight_cf,
            gate_entropy_weight=gate_entropy_weight,
        )

        layer_sizes = self.layer_options[fixed_backbone["layers"]](self.TOTAL_ITEMS)

        model, train_loader, val_loader = self._build_model_and_loaders(
            GHC2F,
            layer_sizes,
            fixed_backbone["nl_type"],
            fixed_backbone["dropout"],
            fixed_backbone["lr"],
            fixed_backbone["tied_weights"],
            ghc2f_kwargs,
        )
        return self._run_and_cleanup(model, train_loader, val_loader, trial=trial)

    def run_loss_weight_study(self, fixed_backbone, n_trials=50):
        pruner = optuna.pruners.SuccessiveHalvingPruner(min_resource=3, reduction_factor=3)
        study = optuna.create_study(direction="maximize", pruner=pruner)
        study.optimize(lambda trial: self.objective_loss_weights(trial, fixed_backbone), n_trials=n_trials)
        print("\n[Stage 2] Best GHC2F loss weights:")
        print(study.best_params)
        return study, study.best_params

    # ------------------------------------------------------------------
    # Stage 3: full-data confirmation of the top-K Stage 1 configs.
    # The Optuna search runs on a 50%/60-70% subsample; the winning config
    # there isn't guaranteed to remain the winner on the full, far sparser
    # dataset. This retrains the top-K candidates on 100% of the data with a
    # real training budget and reports the final ranking, so the reported
    # "best" architecture is validated rather than just search-selected.
    # ------------------------------------------------------------------
    def confirm_on_full_data(self, model_cls, candidate_params_list, num_epochs=STAGE3_EPOCHS):
        self._resample(1.0)
        results = []
        for i, params in enumerate(candidate_params_list):
            print(f"\n[Stage 3] Confirming candidate {i + 1}/{len(candidate_params_list)} on 100% data: {params}")
            layer_sizes = self.layer_options[params["layers"]](self.TOTAL_ITEMS)
            extra_kwargs = {k: params[k] for k in STAGE1_GHC2F_DEFAULTS if k in params} if model_cls is GHC2F else {}

            model, train_loader, val_loader = self._build_model_and_loaders(
                model_cls, layer_sizes, params["nl_type"], params["dropout"], params["lr"],
                params["tied_weights"], extra_kwargs,
            )
            hit_rate = self._run_and_cleanup(model, train_loader, val_loader, num_epochs=num_epochs)
            results.append({**params, "full_data_val_hit_rate": hit_rate})

        results_df = pd.DataFrame(results).sort_values("full_data_val_hit_rate", ascending=False)
        print("\n[Stage 3] Full-data confirmation ranking:")
        print(results_df[["layers", "tied_weights", "full_data_val_hit_rate"]])
        return results_df


def top_k_trial_params(study, k):
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    completed.sort(key=lambda t: t.value, reverse=True)
    return [t.params for t in completed[:k]]


def get_stage2_sample_frac(dataset_name: str) -> float:
    """Rotten Tomatoes has the fewest users after preprocessing (~2.6k) and is
    where GHCF shows its largest gains (Table 3) — so it gets a larger Stage 2
    sample to keep the loss-weight comparison from being dominated by noise.
    Raised across the board (0.3/0.45 -> 0.6/0.7) now that per-epoch cost
    tracks real LOOCV-eligible users rather than TOTAL_USERS — see
    STAGE1_SAMPLE_FRAC."""
    if "rotten" in dataset_name.lower():
        return 0.7
    return 0.6


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, help="Dataset name (matches datasets/{name}.csv)")
    parser.add_argument("--model", type=str, required=True, choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--stage1_trials", type=int, default=60)
    parser.add_argument("--stage2_trials", type=int, default=50)
    parser.add_argument("--skip_stage3", action="store_true", help="Skip the full-data confirmation stage")
    parser.add_argument("--out_dir", type=str, default="grid_search/best_hyperparameters")
    parser.add_argument("--batch_size", type=int, default=2048,
                         help="DataLoader batch size. Batches/epoch = TOTAL_USERS/batch_size "
                              "regardless of sample_frac, so this is the main per-epoch cost lever.")
    parser.add_argument("--num_workers", type=int, default=0,
                         help="DataLoader worker processes for negative sampling / collate. "
                              "RankingTrainDataset now only visits real LOOCV-eligible users "
                              "(a small population on these datasets, often 1-3 batches/epoch), "
                              "so worker process spawn/IPC overhead tends to cost more than it "
                              "saves — raise this only if profiling shows data loading, not GPU "
                              "compute, is the bottleneck.")
    args = parser.parse_args()

    model_cls = MODEL_REGISTRY[args.model]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Stage 1: backbone search ---
    print(f"=== Stage 1: backbone search for {args.dataset} / {args.model} "
          f"({STAGE1_SAMPLE_FRAC:.0%}) ===")
    stage1 = GatedAEOptimizer(dataset_name=args.dataset, device=device, sample_frac=STAGE1_SAMPLE_FRAC,
                               batch_size=args.batch_size, num_workers=args.num_workers)
    stage1_study, best_backbone = stage1.run_backbone_study(model_cls, n_trials=args.stage1_trials)

    combined_params = dict(best_backbone)
    combined_params["dataset"] = args.dataset
    combined_params["model"] = args.model
    combined_params["stage1_sample_frac"] = 0.2

    os.makedirs(args.out_dir, exist_ok=True)

    # Export full trial history for sensitivity analysis (HR vs. layer tier,
    # tied_weights, lr, etc.) — addresses the "sensitivity to review density /
    # hyperparameters" claim that previously had no supporting figure.
    stage1_study.trials_dataframe().to_csv(
        os.path.join(args.out_dir, f"{args.dataset}_{args.model}_stage1_trials.csv"), index=False
    )

    # --- Stage 2: loss-weight search (GHC2F only), backbone fixed ---
    if model_cls is GHC2F:
        stage2_frac = get_stage2_sample_frac(args.dataset)
        print(f"\n=== Stage 2: loss-weight search for {args.dataset} ({stage2_frac:.0%}) ===")
        stage2 = GatedAEOptimizer(dataset_name=args.dataset, device=device, sample_frac=stage2_frac,
                                   batch_size=args.batch_size, num_workers=args.num_workers)
        stage2_study, best_loss_weights = stage2.run_loss_weight_study(
            fixed_backbone=best_backbone, n_trials=args.stage2_trials
        )

        combined_params.update(best_loss_weights)
        combined_params["stage2_sample_frac"] = stage2_frac

        stage2_study.trials_dataframe().to_csv(
            os.path.join(args.out_dir, f"{args.dataset}_{args.model}_stage2_trials.csv"), index=False
        )

    # --- Stage 3: confirm top-K Stage 1 candidates on 100% data ---
    if not args.skip_stage3:
        print(f"\n=== Stage 3: full-data confirmation for {args.dataset} / {args.model} ===")
        stage3 = GatedAEOptimizer(dataset_name=args.dataset, device=device, sample_frac=1.0,
                                   batch_size=args.batch_size, num_workers=args.num_workers)
        candidates = top_k_trial_params(stage1_study, STAGE3_TOP_K)
        confirmation_df = stage3.confirm_on_full_data(model_cls, candidates)
        confirmation_df.to_csv(
            os.path.join(args.out_dir, f"{args.dataset}_{args.model}_stage3_confirmation.csv"), index=False
        )
        # If the subsample-selected "best" wasn't actually best on full data,
        # use the confirmed winner going forward instead of silently keeping
        # the Stage 1 pick.
        confirmed_best = confirmation_df.iloc[0].to_dict()
        for key in ("layers", "tied_weights", "nl_type", "dropout", "lr"):
            combined_params[key] = confirmed_best[key]
        combined_params["full_data_val_hit_rate"] = confirmed_best["full_data_val_hit_rate"]

    # --- Save ---
    out_path = os.path.join(args.out_dir, f"{args.dataset}_{args.model}.csv")
    pd.DataFrame([combined_params]).to_csv(out_path, index=False)
    print(f"\nSaved combined hyperparameters to {out_path}")