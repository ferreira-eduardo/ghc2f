from functools import partial

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from torch.utils.data import Dataset, DataLoader

from utils.utils import AspectDataset


class RankingTrainDataset(Dataset):
    def __init__(self, user_item_matrix, df_text, df_full):
        self.matrix = user_item_matrix
        self.num_users = user_item_matrix.shape[0]
        self.num_items = user_item_matrix.shape[1]
        self.all_items = np.arange(self.num_items)

        self.user_interactions = df_full.groupby('userId')['itemId'].apply(set).to_dict()

        self.user_text_map = df_text.groupby('userId')
        self.text_cols = [col for col in df_text.columns if str(col).isdigit()]

    def __len__(self):
        return self.num_users

    def __getitem__(self, idx):
        pos_items = list(self.user_interactions.get(idx, []))
        pos_item = np.random.choice(pos_items) if pos_items else 0

        neg_item = np.random.choice(self.all_items)
        while neg_item in self.user_interactions.get(idx, set()):
            neg_item = np.random.choice(self.all_items)

        try:
            u_data = self.user_text_map.get_group(idx)
            user_text = torch.tensor(u_data[self.text_cols].values, dtype=torch.float32)
        except KeyError:
            user_text = torch.zeros((1, len(self.text_cols)), dtype=torch.float32)

        return {
            "user_ids": idx,
            "pos_item_id": pos_item,
            "neg_item_id": neg_item,
            "ratings_in": torch.from_numpy(self.matrix[idx].toarray()).float().squeeze(),
            "user_text": user_text
        }


def train_collate_fn(batch):
    res = {}
    res["user_ids"] = torch.tensor([d["user_ids"] for d in batch])
    res["ratings_in"] = torch.stack([d["ratings_in"] for d in batch])

    res["pos_item_id"] = torch.tensor([d["pos_item_id"] for d in batch], dtype=torch.long)
    res["neg_item_id"] = torch.tensor([d["neg_item_id"] for d in batch], dtype=torch.long)

    topics_list = [d["user_text"] for d in batch]
    res["user_text"] = torch.nn.utils.rnn.pad_sequence(topics_list, batch_first=True)

    lengths = torch.tensor([t.size(0) for t in topics_list])
    max_len = res["user_text"].size(1)
    res["user_mask"] = torch.arange(max_len).expand(len(lengths), max_len) < lengths.unsqueeze(1)

    return res


def loocv_collate_fn(batch, df_full, num_negatives=99):
    """
    Standardizes batch for ranking: 1 positive (already filtered) + N unseen negatives.
    """
    res = {}

    # Standard tensor preparation with leakage protection (ratings_in/tgt handled by Dataset)
    res["user_ids"] = torch.tensor([d["user_ids"] for d in batch])
    res["ratings_in"] = torch.stack([d["ratings_in"].clone().detach() for d in batch])

    # Handle variable length user topics
    topics_list = [d["user_text"].clone().detach() for d in batch]
    res["user_text"] = torch.nn.utils.rnn.pad_sequence(topics_list, batch_first=True)

    lengths = torch.tensor([t.size(0) for t in topics_list])
    max_len = res["user_text"].size(1)
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

def create_sparse_matrix(df, num_users, num_items):
    """Cria matriz esparsa garantindo que ratings sejam float para o AE."""
    return csr_matrix(
        (df['rating'].values.astype(np.float32), (df['userId'], df['itemId'])),
        shape=(num_users, num_items)
    )


def loocv_collate_fn_text(batch, text_embeddings, user_reviews_dict):
    res = {}

    # 1. IDs básicos (ajustado para bater com seu Dataset 'user_id')
    user_ids = torch.tensor([d["user_id"] for d in batch])
    res["user_ids"] = user_ids
    res["target_items"] = torch.tensor([d["pos_item_id"] for d in batch])
    res["ratings_in"] = torch.stack([d["ratings_in"] for d in batch])

    # 2. Embedding do Item Alvo (Positivo)
    # No Dataset, o pos_text_seq já vem pronto para o item alvo da validação
    res["pos_text_seq"] = torch.stack([d["pos_text_seq"] for d in batch])

    # 3. Gerar Histórico do Usuário (A Query para a Atenção)
    user_histories = []
    for u in user_ids.tolist():
        # Buscamos no dicionário de reviews do TREINO
        h_idx = user_reviews_dict.get(u, [])
        if len(h_idx) > 0:
            u_emb = text_embeddings[h_idx].mean(dim=0, keepdim=True)
        else:
            u_emb = torch.zeros((1, text_embeddings.size(1)))
        user_histories.append(u_emb)

    res["user_history_text"] = torch.stack(user_histories)  # [B, 1, 768]

    return res


# for train
def loocv_collate_fn_text_train(batch, text_embeddings, user_reviews_dict):
    # IDs e dados básicos
    user_ids = torch.tensor([d["user_id"] for d in batch])
    pos_item_ids = torch.tensor([d["pos_item_id"] for d in batch])
    neg_item_ids = torch.tensor([d["neg_item_id"] for d in batch])
    ratings_in = torch.stack([d["ratings_in"] for d in batch])

    # Textos (Embeddings)
    pos_text_seq = torch.stack([d["pos_text_seq"] for d in batch])
    neg_text_seq = torch.stack([d["neg_text_seq"] for d in batch])

    # Histórico do Usuário
    user_histories = []
    for u in user_ids.tolist():
        h_idx = user_reviews_dict.get(u, [])
        if len(h_idx) > 0:
            u_emb = text_embeddings[h_idx].mean(dim=0, keepdim=True)
        else:
            u_emb = torch.zeros((1, text_embeddings.size(1)))
        user_histories.append(u_emb)

    return {
        "user_ids": user_ids,
        "pos_item_id": pos_item_ids,
        "neg_item_id": neg_item_ids,
        "ratings_in": ratings_in,
        "pos_text_seq": pos_text_seq,
        "neg_text_seq": neg_text_seq,
        "user_history_text": torch.stack(user_histories)
    }




def generate_data_loaders (train_matrix, train_df, val_df, test_df, text_embedding, batch_size):

    train_reviews_dict = train_df.groupby('userId').groups

    collate_fn_train = partial(
        loocv_collate_fn_text_train,
        text_embeddings=torch.from_numpy(text_embedding).float(),
        user_reviews_dict=train_reviews_dict
    )

    train_loader = DataLoader(
        AspectDataset(train_matrix, train_df, text_embedding),
        batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn_train, num_workers=4
    )

    collate_val = partial(
        loocv_collate_fn_text_train,
        text_embeddings=torch.from_numpy(text_embedding).float(),
        user_reviews_dict=train_reviews_dict
    )

    val_loader = DataLoader(
        AspectDataset(train_matrix, val_df, text_embedding),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_val,
        num_workers=4
    )

    test_relevant = test_df[test_df["is_relevant"] == True].copy()
    history_df = pd.concat([train_df, val_df])
    history_reviews_dict = history_df.groupby('userId').groups
    collate_test = partial(
        loocv_collate_fn_text_train,
        text_embeddings=torch.from_numpy(text_embedding).float(),
        user_reviews_dict=history_reviews_dict
    )

    test_loader = DataLoader(
        AspectDataset(train_matrix, test_relevant, text_embedding),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_test,
        num_workers=4
    )


    return train_loader, val_loader, test_loader