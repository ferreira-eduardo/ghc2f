import numpy as np
import torch
from scipy.sparse import csr_matrix
from torch.utils.data import Dataset

class RankingTrainDataset(Dataset):
    def __init__(self, user_item_matrix, df_topics, df_full):
        self.matrix = user_item_matrix
        self.num_users = user_item_matrix.shape[0]
        self.num_items = user_item_matrix.shape[1]
        self.all_items = np.arange(self.num_items)

        self.user_interactions = df_full.groupby('userId')['itemId'].apply(set).to_dict()

        self.user_topics_map = df_topics.groupby('userId')
        self.topic_cols = [col for col in df_topics.columns if col.isdigit()]

    def __len__(self):
        return self.num_users

    def __getitem__(self, idx):
        pos_items = list(self.user_interactions.get(idx, []))
        pos_item = np.random.choice(pos_items) if pos_items else 0

        neg_item = np.random.choice(self.all_items)
        while neg_item in self.user_interactions.get(idx, set()):
            neg_item = np.random.choice(self.all_items)

        try:
            u_data = self.user_topics_map.get_group(idx)
            user_topics = torch.tensor(u_data[self.topic_cols].values, dtype=torch.float32)
        except KeyError:
            user_topics = torch.zeros((1, len(self.topic_cols)), dtype=torch.float32)

        return {
            "user_ids": idx,
            "pos_item_id": pos_item,
            "neg_item_id": neg_item,
            "ratings_in": torch.from_numpy(self.matrix[idx].toarray()).float().squeeze(),
            "user_topics": user_topics
        }


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

def create_sparse_matrix(df, num_users, num_items):
    """Cria matriz esparsa garantindo que ratings sejam float para o AE."""
    return csr_matrix(
        (df['rating'].values.astype(np.float32), (df['userId'], df['itemId'])),
        shape=(num_users, num_items)
    )


def create_gpu_sparse_matrix(df, total_users, total_items, device):
    indices = torch.stack([
        torch.from_numpy(df['userId'].values.copy()).long(),
        torch.from_numpy(df['itemId'].values.copy()).long()
    ])
    values = torch.from_numpy(df['rating'].values).float()

    return torch.sparse_coo_tensor(indices, values, (total_users, total_items)).to(device)

class LOOCVCollateWrapper:
    def __init__(self, val_df):
        self.val_df = val_df

    def __call__(self, batch):
        return loocv_collate_fn(batch, self.val_df)