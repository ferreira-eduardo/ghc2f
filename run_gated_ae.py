import gc

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from model.ghc2f import GHC2F

from utils.leave_one_out_cv import get_loocv_fold_normalized
from utils.train_models import train_model

def loocv_collate_fn(batch, df_full, num_negatives=99):
    """
    Standardizes batch for ranking: 1 positive (already filtered) + N unseen negatives.
    """
    res = {}

    # Standard tensor preparation with leakage protection (ratings_in/tgt handled by Dataset)
    res["user_ids"] = torch.tensor([d["user_ids"] for d in batch])
    res["ratings_in"] = torch.stack([d["ratings_in"].clone().detach() for d in batch])

    # Handle variable length user topics
    topics_list = [d["user_topics"].clone().detach() for d in batch]
    res["user_topics"] = torch.nn.utils.rnn.pad_sequence(topics_list, batch_first=True)

    lengths = torch.tensor([t.size(0) for t in topics_list])
    max_len = res["user_topics"].size(1)
    res["user_mask"] = torch.arange(max_len).expand(len(lengths), max_len) < lengths.unsqueeze(1)

    test_items = []
    neg_items = []
    for d in batch:
        u_id = d["user_ids"]
        # "Unseen" = Items the user has NEVER interacted with in the whole datasets
        test_items.append(d["pos_item_id"])
        interacted = set(df_full[df_full['userId'] == u_id]['itemId'].unique())
        candidates = np.setdiff1d(np.arange(df_full.itemId.max() + 1), list(interacted))

        # Randomly sample 99 items to compose the evaluation
        if len(candidates) >= num_negatives:
            negs = np.random.choice(candidates, num_negatives, replace=False)
        else:
            # Fallback if the user has seen almost everything
            negs = np.random.choice(candidates, num_negatives, replace=True)

        neg_items.append(torch.tensor(negs, dtype=torch.long))

    res["pos_item_id"] = torch.tensor(test_items)
    res["neg_item_id"] = torch.stack(neg_items)

    return res


def train_collate_fn(batch):
    res = {}
    res["user_ids"] = torch.tensor([d["user_ids"] for d in batch])
    res["ratings_in"] = torch.stack([d["ratings_in"] for d in batch])

    res["pos_item_id"] = torch.tensor([d["pos_item_id"] for d in batch], dtype=torch.long)
    res["neg_item_id"] = torch.tensor([d["neg_item_id"] for d in batch], dtype=torch.long)

    topics_list = [d["user_topics"] for d in batch]
    res["user_topics"] = torch.nn.utils.rnn.pad_sequence(topics_list, batch_first=True)

    lengths = torch.tensor([t.size(0) for t in topics_list])
    max_len = res["user_topics"].size(1)
    res["user_mask"] = torch.arange(max_len).expand(len(lengths), max_len) < lengths.unsqueeze(1)

    return res

class RankingTrainDataset(Dataset):
    def __init__(self, user_item_matrix, df_topics, target_df, df_full_for_negs, is_train=False):
        self.matrix = user_item_matrix
        self.num_items = user_item_matrix.shape[1]
        self.is_train = is_train
        self.samples = target_df[['userId', 'itemId']].values.tolist()

        topic_cols = [col for col in df_topics.columns if col.isdigit()]
        self.user_topics_tensor = torch.zeros((user_item_matrix.shape[0], len(topic_cols)))
        for u_id, group in df_topics.groupby('userId'):
            if u_id < self.user_topics_tensor.size(0):
                self.user_topics_tensor[int(u_id)] = torch.tensor(group[topic_cols].values[0], dtype=torch.float32)

        full_interactions = df_full_for_negs.groupby('userId')['itemId'].apply(set).to_dict()
        all_item_indices = np.arange(self.num_items)

        self.neg_candidates = {}
        for u_id in range(user_item_matrix.shape[0]):
            interacted = full_interactions.get(u_id, set())
            self.neg_candidates[u_id] = list(np.setdiff1d(all_item_indices, list(interacted)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        u_id, pos_item = self.samples[idx]
        u_id, pos_item = int(u_id), int(pos_item)

        res = {
            "user_ids": u_id,
            "pos_item_id": pos_item,
            "ratings_in": torch.from_numpy(self.matrix[u_id].toarray()).float().squeeze(),
            "user_topics": self.user_topics_tensor[u_id],
            "neg_candidates": self.neg_candidates.get(u_id, [])  # Pass candidates to collate
        }

        if self.is_train:
            neg_item = np.random.choice(self.neg_candidates[u_id])
            res["neg_item_id"] = neg_item

        return res

def create_sparse_matrix(df, num_users, num_items):
    """Cria matriz esparsa garantindo que ratings sejam float para o AE."""
    return csr_matrix(
        (df['rating'].values.astype(np.float32), (df['userId'], df['itemId'])),
        shape=(num_users, num_items)
    )

def prepare_inputs(df, entity_col, topic_col_names):
    vectors = df[topic_col_names].values

    # Create a temporary series to group
    temp_df = pd.DataFrame({
        entity_col: df[entity_col],
        'vec': list(vectors)
    })

    # Group by User/Item
    grouped = temp_df.groupby(entity_col)['vec'].apply(list).reset_index()

    # Extract IDs
    ids = torch.tensor(grouped[entity_col].values, dtype=torch.long)

    # Convert lists to tensors and pad
    topic_tensors = [torch.tensor(np.array(t), dtype=torch.float32) for t in grouped['vec']]
    topics_padded = pad_sequence(topic_tensors, batch_first=True, padding_value=0.0)

    # Create Mask
    lengths = torch.tensor([len(t) for t in topic_tensors])
    max_len = topics_padded.size(1)
    mask = torch.arange(max_len).expand(len(lengths), max_len) < lengths.unsqueeze(1)

    return ids, topics_padded, mask

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    datasets = ['amazon']
    path = 'datasets/{}.csv'
    num_epochs = 1
    all_results = []
    k_folds = [0, 1, 2, 3, 4]

    for dataset in datasets:
        print(f"\nLoading datasets: {dataset}")
        df = pd.read_csv(path.format(dataset))
        df_topics = pd.read_csv(path.format('topic_dist_' + dataset))


        # # Tipagem para economia de memória e compatibilidade
        df["userId"] = df["userId"].astype(int)
        df["itemId"] = df["itemId"].astype(int)
        df_topics["userId"] = df_topics["userId"].astype(int)
        df_topics["itemId"] = df_topics["itemId"].astype(int)

        TOTAL_ITEMS = df.itemId.max() + 1
        TOTAL_USERS = df.userId.max() + 1
        topic_cols = [col for col in df_topics.columns if col.isdigit()]
        topics_dim = len(topic_cols)

        for fold in k_folds:
            print(f"\n########## FOLD {fold} ##########")
            train, val, test = get_loocv_fold_normalized(df, fold)

            model = GHC2F(
                layer_sizes=[TOTAL_ITEMS, 4096],
                num_users=TOTAL_USERS,
                num_items=TOTAL_ITEMS,
                topics_dim=topics_dim,
                topics_latent_dim=64,
                nl_type="selu",
                dp_drop_prob=0.5,
                learn_rate=1e-4
            ).to(device)

            # Configuração dos perfis de itens globais
            df_topics_train = df_topics[df_topics["itemId"].isin(train["itemId"].unique())].copy()
            item_ids, item_topics, item_mask = prepare_inputs(df_topics_train, "itemId", topic_cols)

            with torch.no_grad():
                profiles = model.item_profiler(item_ids.to(device), item_topics.to(device), item_mask.to(device)).detach()

            full_i_global = torch.zeros((TOTAL_ITEMS, topics_dim), device=device)
            full_i_global[item_ids] = profiles
            model.item_global_profiles = full_i_global

            # Matrizes de Entrada (ratings_in)
            train_matrix = create_sparse_matrix(train, TOTAL_USERS, TOTAL_ITEMS)
            # Para validação e teste, o modelo "enxerga" o histórico de treino para prever o próximo
            history_matrix = train_matrix

            # Dataloader de Treino (Contrastive)
            train_loader = DataLoader(
                RankingTrainDataset(train_matrix, df_topics, train, df, is_train=True),  # Usa triplas de treino
                batch_size=512, shuffle=True, collate_fn=train_collate_fn,
            )

            # Dataloader de Validação (Contrastive)
            val_loader = DataLoader(
                RankingTrainDataset(train_matrix, df_topics, val, df),
                batch_size=64, shuffle=False, collate_fn=lambda x: loocv_collate_fn(x, train)
            )
            test_relevant = test[test["is_relevant"] == True].copy()

            history_for_test = pd.concat([train, val])
            test_loader_ranking = DataLoader(
                RankingTrainDataset(history_matrix, df_topics, test_relevant, df),
                batch_size=64, shuffle=False,
                collate_fn=lambda x: loocv_collate_fn(x, history_for_test),  # 1 pos + 99 negs
            )

            print('Starting training process (BPR Loss)...')
            best_val_loss, train_losses = train_model(model, num_epochs, train_loader, val_loader)

            print("Iniciando Avaliação do Ranking..")
            values_rank = model.evaluate(test_loader_ranking)

            all_results.append({
                "datasets": dataset, "fold": fold,
                **values_rank
            })

            # print(all_results)
            torch.cuda.empty_cache()
            gc.collect()

            # salva cada datasets
            df_results = pd.DataFrame(all_results)
            df_results.to_csv(f"contrastive_bpr_{dataset}.csv", index=False)

        df_results = pd.DataFrame(all_results)
        df_results.to_csv(f"contrastive_bpr_{dataset}.csv", index=False)


if __name__ == "__main__":
    main()