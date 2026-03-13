import optuna
import pandas as pd
import torch
import gc
from torch.utils.data import DataLoader

from utils.ae_utils import prepare_inputs
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

class GatedAEOptimizer:
    def __init__(self, dataset_name, path_template, device):

        self.dataset_name = dataset_name
        self.path_template = path_template
        self.device = device

        # Carregamento e Amostragem de 20%
        print(f"Carregando e amostrando 20% do datasets {dataset_name}...")
        full_df = pd.read_csv(path_template.format(dataset_name))

        # Amostragem por usuários para manter a integridade do problema de recomendação
        unique_users = full_df['userId'].unique()
        sampled_users = pd.Series(unique_users).sample(frac=0.2, random_state=42)
        self.df = full_df[full_df['userId'].isin(sampled_users)].copy()

        self.df_topics = pd.read_csv(path_template.format('topic_dist_' + dataset_name))

        self.topic_cols = [col for col in self.df_topics.columns if col.isdigit()]
        self.topics_dim = len(self.topic_cols)
        # Ajuste de tipos
        self.df["userId"] = self.df["userId"].astype("int32")
        self.df["itemId"] = self.df["itemId"].astype("int32")
        self.TOTAL_ITEMS = full_df.itemId.max() + 1
        self.TOTAL_USERS = full_df.userId.max() + 1

    def objective(self, trial):
        # Definir o Espaço de Busca
        dropout = trial.suggest_categorical("dropout", [0.2, 0.5])
        lr = trial.suggest_categorical("lr", [1e-4, 1e-3])
        nl_type = trial.suggest_categorical("nl_type", ["selu", "relu"])

        # Escolha da arquitetura de layers
        layer_option = trial.suggest_categorical("layers", ["small", "medium", "large", "xl", "exl"])
        if layer_option == "small":
            layer_sizes = [self.TOTAL_ITEMS, 4096]
        elif layer_option == "medium":
            layer_sizes = [self.TOTAL_ITEMS, 4096, 2048]
        elif layer_option == 'large':
            layer_sizes = [self.TOTAL_ITEMS, 4096, 2048, 1024]
        elif layer_option == 'xl':
            layer_sizes = [self.TOTAL_ITEMS, 4096, 2048, 1024, 512]
        else:
            layer_sizes = [self.TOTAL_ITEMS, 4096, 2048, 1024, 512, 256]

        # Setup do Fold (Usando apenas 1 fold para otimização rápida)
        train, val, _ = get_loocv_fold_normalized(self.df, 0)

        model = GHC2F(
            layer_sizes=layer_sizes,
            num_users=self.TOTAL_USERS,
            num_items=self.TOTAL_ITEMS,
            topics_dim=self.topics_dim,
            topics_latent_dim=64,
            nl_type=nl_type,
            dp_drop_prob=dropout,
            learn_rate=lr
        ).to(self.device)

        # Preparação dos Perfis (Mesma lógica do seu script)
        df_topics_train = self.df_topics[self.df_topics["itemId"].isin(train["itemId"].unique())].copy()
        item_ids, item_topics, item_mask = prepare_inputs(df_topics_train, "itemId", self.topic_cols)

        with torch.no_grad():
            profiles = model.item_profiler(item_ids.to(self.device), item_topics.to(self.device),
                                           item_mask.to(self.device)).detach()

        full_i_global = torch.zeros((self.TOTAL_ITEMS, self.topics_dim), device=self.device)
        full_i_global[item_ids] = profiles
        model.item_global_profiles = full_i_global

        # DataLoaders
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


        best_val_loss, _ = train_model(model, num_epochs=15, train_loader=train_loader, val_loader=val_loader)

        del model
        torch.cuda.empty_cache()
        gc.collect()

        return best_val_loss  # Optuna tentará minimizar este valor

    def run_study(self, n_trials=20):
        study = optuna.create_study(direction="minimize")

        study.optimize(self.objective, n_trials=n_trials)

        print("\nMelhores Hiperparâmetros:")
        print(study.best_params)
        return study.best_params


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python optimizer_hyper.py <nome_do_dataset>")
        sys.exit(1)

    target_dataset = sys.argv[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    path = 'datasets/{}.csv'

    print(f"Iniciando otimização para: {target_dataset}")
    optimizer = GatedAEOptimizer(dataset_name=target_dataset, path_template=path, device=device)
    best_params = optimizer.run_study(n_trials=30)

    # Salva o resultado
    df_best = pd.DataFrame([best_params])
    df_best['dataset'] = target_dataset
    df_best.to_csv(f"best_params_{target_dataset}.csv", index=False)
