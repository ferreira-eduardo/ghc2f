import optuna
import pandas as pd
import torch
import gc
from torch.utils.data import DataLoader

from utils.utils import prepare_inputs
from utils.dataset_utils import RankingTrainDataset, train_collate_fn, create_sparse_matrix
from utils.dataset_utils import loocv_collate_fn
from model.ghc2f import GHC2F
from utils.leave_one_out_cv import get_loocv_fold_normalized
from utils.train_model import train_model

import sys


class LOOCVCollateWrapper:
    def __init__(self, val_df):
        self.val_df = val_df

    def __call__(self, batch):
        return loocv_collate_fn(batch, self.val_df)


LAYER_OPTIONS = {
    "small": lambda n: [n, 4096],
    "medium": lambda n: [n, 4096, 2048],
    "large": lambda n: [n, 4096, 2048, 1024],
    "xl": lambda n: [n, 4096, 2048, 1024, 512],
    "exl": lambda n: [n, 4096, 2048, 1024, 512, 256],
}

# Defaults used for GHC2F-specific loss hyperparameters during Stage 1
# (backbone search), so architecture/optimizer choices aren't confounded by
# an untuned loss-weight configuration.

STAGE1_GHC2F_DEFAULTS = dict(
    semantic_gamma=0.5,
    contrastive_target="text",
    stop_grad_cf=True,
    dual_head_decoder=True,
    cl_weight=0.1,
    reg_weight=1e-5,
    mmse_weight_fused=1.0,
    mmse_weight_cf=1.0,
    gate_entropy_weight=0.0,
)


class GatedAEOptimizer:
    def __init__(self, dataset_name, path_template, device, sample_frac=0.2, random_state=42):
        self.dataset_name = dataset_name
        self.path_template = path_template
        self.device = device
        self.random_state = random_state

        print(f"Loading {dataset_name} (samples of {sample_frac:.0%} users)...")
        self.full_df = pd.read_csv(path_template.format(dataset_name))
        self._resample(sample_frac)

        self.df_topics = pd.read_csv(path_template.format('topic_dist_' + dataset_name))
        self.topic_cols = [col for col in self.df_topics.columns if col.isdigit()]
        self.topics_dim = len(self.topic_cols)

        self.TOTAL_ITEMS = self.full_df.itemId.max() + 1
        self.TOTAL_USERS = self.full_df.userId.max() + 1

    def _resample(self, frac: float):
        """Re-sample users at a given fraction, keyed by the same random_state
        so Stage 1 and Stage 2 samples are reproducible independently."""
        unique_users = self.full_df['userId'].unique()
        sampled_users = pd.Series(unique_users).sample(frac=frac, random_state=self.random_state)
        self.df = self.full_df[self.full_df['userId'].isin(sampled_users)].copy()
        self.df["userId"] = self.df["userId"].astype("int32")
        self.df["itemId"] = self.df["itemId"].astype("int32")

    def _build_model_and_loaders(self, layer_sizes, nl_type, dropout, lr, ghc2f_kwargs):
        """Shared setup: fold split, model construction, item profiles, dataloaders.
        Used by both the backbone objective and the loss-weight objective so the
        two stages stay consistent in everything except what they search over."""
        train, val, _ = get_loocv_fold_normalized(self.df, 0)

        model = GHC2F(
            layer_sizes=layer_sizes,
            num_users=self.TOTAL_USERS,
            num_items=self.TOTAL_ITEMS,
            text_dim=self.topics_dim,
            text_latent_dim=64,
            nl_type=nl_type,
            dp_drop_prob=dropout,
            learn_rate=lr,
            **ghc2f_kwargs,
        ).to(self.device)

        df_topics_train = self.df_topics[self.df_topics["itemId"].isin(train["itemId"].unique())].copy()
        item_ids, item_topics, item_mask = prepare_inputs(df_topics_train, "itemId", self.topic_cols)

        with torch.no_grad():
            profiles = model.item_profiler(
                item_ids.to(self.device), item_topics.to(self.device), item_mask.to(self.device)
            ).detach()

        full_i_global = torch.zeros((self.TOTAL_ITEMS, self.topics_dim), device=self.device)
        full_i_global[item_ids] = profiles
        model.item_global_profiles = full_i_global

        train_matrix = create_sparse_matrix(train, self.TOTAL_USERS, self.TOTAL_ITEMS)
        train_loader = DataLoader(
            RankingTrainDataset(train_matrix, self.df_topics, train, self.df, self.device, is_train=True),
            batch_size=512, shuffle=True, collate_fn=train_collate_fn, num_workers=0
        )

        val_collate = LOOCVCollateWrapper(val)
        val_loader = DataLoader(
            RankingTrainDataset(train_matrix, self.df_topics, val, self.df, self.device),
            batch_size=512, shuffle=False, collate_fn=val_collate, num_workers=0
        )

        return model, train_loader, val_loader

    def _run_and_cleanup(self, model, train_loader, val_loader, num_epochs=15):
        best_val_loss, _ = train_model(model, num_epochs=num_epochs, train_loader=train_loader, val_loader=val_loader)

        del model
        torch.cuda.empty_cache()
        gc.collect()

        return best_val_loss

    # ------------------------------------------------------------------
    # Stage 1: backbone / optimizer hyperparameters
    # ------------------------------------------------------------------
    def objective_backbone(self, trial):
        dropout = trial.suggest_categorical("dropout", [0.2, 0.5])
        lr = trial.suggest_categorical("lr", [1e-4, 1e-3])
        nl_type = trial.suggest_categorical("nl_type", ["selu", "relu"])
        layer_option = trial.suggest_categorical("layers", list(LAYER_OPTIONS.keys()))

        layer_sizes = LAYER_OPTIONS[layer_option](self.TOTAL_ITEMS)

        model, train_loader, val_loader = self._build_model_and_loaders(
            layer_sizes, nl_type, dropout, lr, STAGE1_GHC2F_DEFAULTS
        )
        return self._run_and_cleanup(model, train_loader, val_loader)

    def run_backbone_study(self, n_trials=20):
        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective_backbone, n_trials=n_trials)
        print("\n[Stage 1] Melhores hiperparâmetros de backbone:")
        print(study.best_params)
        return study.best_params

    # ------------------------------------------------------------------
    # Stage 2: GHC2F-specific loss-weight hyperparameters
    # (backbone fixed to the Stage 1 winner)
    # ------------------------------------------------------------------
    def objective_loss_weights(self, trial, fixed_backbone):
        topic_gamma = trial.suggest_float("topic_gamma", 0.1, 0.9)

        contrastive_target = trial.suggest_categorical(
            "contrastive_target", ["text", "collaborative"]
        )
        # No-op unless contrastive_target="collaborative"; sampled unconditionally
        # to keep the search space shape fixed across trials.
        stop_grad_cf = trial.suggest_categorical("stop_grad_cf", [True, False])

        dual_head_decoder = trial.suggest_categorical("dual_head_decoder", [True, False])

        cl_weight = trial.suggest_float("cl_weight", 1e-3, 1.0, log=True)
        reg_weight = trial.suggest_float("reg_weight", 1e-6, 1e-3, log=True)

        # mmse_weight_cf only matters when dual_head_decoder=True; same
        # fixed-shape rationale as stop_grad_cf above.
        mmse_weight_fused = trial.suggest_float("mmse_weight_fused", 0.1, 2.0)
        mmse_weight_cf = trial.suggest_float("mmse_weight_cf", 0.0, 2.0)

        gate_entropy_weight = trial.suggest_float("gate_entropy_weight", 0.0, 0.1)

        ghc2f_kwargs = dict(
            topic_gamma=topic_gamma,
            contrastive_target=contrastive_target,
            stop_grad_cf=stop_grad_cf,
            dual_head_decoder=dual_head_decoder,
            cl_weight=cl_weight,
            reg_weight=reg_weight,
            mmse_weight_fused=mmse_weight_fused,
            mmse_weight_cf=mmse_weight_cf,
            gate_entropy_weight=gate_entropy_weight,
        )

        layer_sizes = LAYER_OPTIONS[fixed_backbone["layers"]](self.TOTAL_ITEMS)

        model, train_loader, val_loader = self._build_model_and_loaders(
            layer_sizes,
            fixed_backbone["nl_type"],
            fixed_backbone["dropout"],
            fixed_backbone["lr"],
            ghc2f_kwargs,
        )
        return self._run_and_cleanup(model, train_loader, val_loader)

    def run_loss_weight_study(self, fixed_backbone, n_trials=30):
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: self.objective_loss_weights(trial, fixed_backbone), n_trials=n_trials)
        print("\n[Stage 2] Melhores pesos de perda do GHC2F:")
        print(study.best_params)
        return study.best_params


def get_stage2_sample_frac(dataset_name: str) -> float:
    """Rotten Tomatoes has the fewest users after preprocessing (~2.6k) and is
    where GHCF shows its largest gains (Table 3) — so it gets a larger Stage 2
    sample to keep the loss-weight comparison from being dominated by noise."""
    if "rotten" in dataset_name.lower():
        return 0.45
    return 0.35


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python optimizer_hyper.py <nome_do_dataset>")
        sys.exit(1)

    target_dataset = sys.argv[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path = 'datasets/{}.csv'

    # --- Stage 1: backbone search at 20% ---
    print(f"=== Stage 1: busca de backbone para {target_dataset} (20%) ===")
    stage1 = GatedAEOptimizer(dataset_name=target_dataset, path_template=path, device=device, sample_frac=0.2)
    best_backbone = stage1.run_backbone_study(n_trials=20)

    # --- Stage 2: loss-weight search at a higher fraction, backbone fixed ---
    stage2_frac = get_stage2_sample_frac(target_dataset)
    print(f"\n=== Stage 2: busca de pesos de perda para {target_dataset} ({stage2_frac:.0%}) ===")
    stage2 = GatedAEOptimizer(dataset_name=target_dataset, path_template=path, device=device, sample_frac=stage2_frac)
    best_loss_weights = stage2.run_loss_weight_study(fixed_backbone=best_backbone, n_trials=30)

    # --- Combine and save ---
    combined_params = {**best_backbone, **best_loss_weights}
    combined_params["dataset"] = target_dataset
    combined_params["stage1_sample_frac"] = 0.2
    combined_params["stage2_sample_frac"] = stage2_frac

    df_best = pd.DataFrame([combined_params])
    df_best.to_csv(f"best_params_{target_dataset}.csv", index=False)
    print(f"\nHiperparâmetros combinados salvos em best_params_{target_dataset}.csv")