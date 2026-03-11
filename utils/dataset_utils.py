import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

class RankingTrainDataset(Dataset):
    def __init__(self, user_item_matrix, df_topics, target_df, df_full_for_negs, is_train=False):
        self.matrix = user_item_matrix
        self.num_items = user_item_matrix.shape[1]
        self.is_train = is_train
        self.samples = target_df[['userId', 'itemId']].values.tolist()

        # 1. Pre-compute User Topics Matrix (O(1) Access)
        topic_cols = [col for col in df_topics.columns if col.isdigit()]
        self.user_topics_tensor = torch.zeros((user_item_matrix.shape[0], len(topic_cols)))
        for u_id, group in df_topics.groupby('userId'):
            if u_id < self.user_topics_tensor.size(0):
                self.user_topics_tensor[int(u_id)] = torch.tensor(group[topic_cols].values[0], dtype=torch.float32)

        # 2. Pre-compute Negative Candidates for LOOCV (Critical for speed)
        # Find all items each user has EVER interacted with
        full_interactions = df_full_for_negs.groupby('userId')['itemId'].apply(set).to_dict()
        all_item_indices = np.arange(self.num_items)

        self.neg_candidates = {}
        for u_id in range(user_item_matrix.shape[0]):
            interacted = full_interactions.get(u_id, set())
            # Store as list for fast random choice later
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
            # Fast BPR negative sampling
            neg_item = np.random.choice(self.neg_candidates[u_id])
            res["neg_item_id"] = neg_item

        return res


def train_collate_fn(batch):
    res = {}
    res["user_ids"] = torch.tensor([d["user_ids"] for d in batch])
    res["ratings_in"] = torch.stack([d["ratings_in"] for d in batch])

    # Pegamos o positivo (Z-score alto) e o negativo (Z-score baixo)
    # que o seu Dataset já identificou
    res["pos_item_id"] = torch.tensor([d["pos_item_id"] for d in batch], dtype=torch.long)
    res["neg_item_id"] = torch.tensor([d["neg_item_id"] for d in batch], dtype=torch.long)

    # Padding de tópicos (mantendo sua lógica original)
    topics_list = [d["user_topics"] for d in batch]
    res["user_topics"] = torch.nn.utils.rnn.pad_sequence(topics_list, batch_first=True)

    lengths = torch.tensor([t.size(0) for t in topics_list])
    max_len = res["user_topics"].size(1)
    res["user_mask"] = torch.arange(max_len).expand(len(lengths), max_len) < lengths.unsqueeze(1)

    return res


def loocv_collate_fn(batch, df_full, num_negatives=99):
    res = {}

    res["user_ids"] = torch.tensor([d["user_ids"] for d in batch])

    # Stack the input ratings
    ratings_in = torch.stack([d["ratings_in"].clone().detach() for d in batch])

    # MASKING
    for i, d in enumerate(batch):
        target_item = d["pos_item_id"]
        ratings_in[i, target_item] = 0

    res["ratings_in"] = ratings_in

    topics_list = [d["user_topics"].clone().detach() for d in batch]
    res["user_topics"] = torch.nn.utils.rnn.pad_sequence(topics_list, batch_first=True)
    lengths = torch.tensor([t.size(0) for t in topics_list])
    max_len = res["user_topics"].size(1)
    res["user_mask"] = torch.arange(max_len).expand(len(lengths), max_len) < lengths.unsqueeze(1)

    test_items = []
    neg_items = []

    max_item_id = df_full.itemId.max()

    for d in batch:
        u_id = d["user_ids"]
        test_items.append(d["pos_item_id"])

        interacted = set(df_full[df_full['userId'] == u_id]['itemId'].unique())

        candidates = np.setdiff1d(np.arange(max_item_id + 1), list(interacted))

        if len(candidates) >= num_negatives:
            negs = np.random.choice(candidates, num_negatives, replace=False)
        else:
            negs = np.random.choice(candidates, num_negatives, replace=True)

        neg_items.append(torch.tensor(negs, dtype=torch.long))

    res["pos_item_id"] = torch.tensor(test_items)
    res["neg_item_id"] = torch.stack(neg_items)

    return res