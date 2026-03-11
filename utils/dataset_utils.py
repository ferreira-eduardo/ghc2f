import numpy as np
import torch
from scipy.sparse import csr_matrix
from torch.utils.data import Dataset

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
        torch.from_numpy(df['userId'].values).long(),
        torch.from_numpy(df['itemId'].values).long()
    ])
    values = torch.from_numpy(df['rating'].values).float()

    return torch.sparse_coo_tensor(indices, values, (total_users, total_items)).to(device)