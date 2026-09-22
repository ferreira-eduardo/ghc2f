import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from torch.utils.data import Dataset


class RankingTrainDataset(Dataset):
    def __init__(self, user_item_matrix, df_text, df_full, aspect_probs=None, aspect_mask=None, seed=None):
        """
        aspect_probs, aspect_mask : Optional[np.ndarray]
            Dense, per-user arrays row-aligned by userId (see
            aspects/build_user_aspect_profiles.py), shapes [TOTAL_USERS, 3*K]
            and [TOTAL_USERS, K]. None (default) omits the "aspect_probs" /
            "aspect_mask" batch keys entirely -- only SemanticDisentangledAE
            needs them; every other model ignores unknown batch keys, but
            there's no reason to build/carry these arrays when nothing will
            read them.
        """
        self.matrix = user_item_matrix
        self.num_items = user_item_matrix.shape[1]
        self.all_items = np.arange(self.num_items)
        self.aspect_probs = aspect_probs
        self.aspect_mask = aspect_mask
        self.rng = np.random.default_rng(seed)

        # Iterate only the users actually present in this split (df_full),
        # not every global user id in user_item_matrix. Previously __len__
        # returned user_item_matrix.shape[0] (=TOTAL_USERS) unconditionally,
        # so a user absent from df_full — e.g. filtered out of the LOOCV
        # split for having too few interactions, or simply outside this
        # sample_frac subset — still got a "ghost" row here, defaulting to
        # pos_item=0 and an all-zero ratings_in vector. In sparse datasets
        # that's the overwhelming majority of rows (e.g. only ~0.3% of users
        # survive All_Beauty's LOOCV filter at a 20% sample): every ghost row
        # feeds the model an identical all-zero input, so eval's hit_rate/
        # ndcg/mrr ended up dominated by "what does the model output for a
        # blank input" instead of genuine ranking quality on real users.
        self.user_ids = np.sort(df_full['userId'].unique())

        self.user_interactions = df_full.groupby('userId')['itemId'].apply(set).to_dict()

        self.user_text_map = df_text.groupby('userId')
        self.text_cols = [col for col in df_text.columns if str(col).isdigit()]

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, i):
        idx = int(self.user_ids[i])

        pos_items = list(self.user_interactions.get(idx, []))
        pos_item = self.rng.choice(pos_items) if pos_items else 0

        neg_item = self.rng.choice(self.all_items)
        while neg_item in self.user_interactions.get(idx, set()):
            neg_item = self.rng.choice(self.all_items)

        try:
            u_data = self.user_text_map.get_group(idx)
            user_text = torch.tensor(u_data[self.text_cols].values, dtype=torch.float32)
        except KeyError:
            user_text = torch.zeros((1, len(self.text_cols)), dtype=torch.float32)

        item = {
            "user_ids": idx,
            "pos_item_id": pos_item,
            "neg_item_id": neg_item,
            "ratings_in": torch.from_numpy(self.matrix[idx].toarray()).float().squeeze(),
            "user_text": user_text
        }

        if self.aspect_probs is not None:
            item["aspect_probs"] = torch.from_numpy(self.aspect_probs[idx]).float()
            item["aspect_mask"] = torch.from_numpy(self.aspect_mask[idx]).float()

        return item


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

    if "aspect_probs" in batch[0]:
        res["aspect_probs"] = torch.stack([d["aspect_probs"] for d in batch])
        res["aspect_mask"] = torch.stack([d["aspect_mask"] for d in batch])

    return res


def build_interacted_by_user(df_full):
    """Precompute {userId: set(itemId)} once per fold, so loocv_collate_fn
    doesn't re-filter the whole dataframe for every sample in every batch."""
    return df_full.groupby('userId')['itemId'].apply(set).to_dict()


def loocv_collate_fn(batch, interacted_by_user, num_items, num_negatives=99):
    """
    Standardizes batch for ranking: 1 positive (already filtered) + N unseen negatives.

    interacted_by_user: {userId: set(itemId)}, built once per fold via
    build_interacted_by_user (see GatedAEOptimizer._get_fold_cache) instead
    of being rebuilt from the raw dataframe on every call.
    num_items: total (global) item count — the candidate universe.
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

    if "aspect_probs" in batch[0]:
        res["aspect_probs"] = torch.stack([d["aspect_probs"].clone().detach() for d in batch])
        res["aspect_mask"] = torch.stack([d["aspect_mask"].clone().detach() for d in batch])

    all_items = np.arange(num_items)
    test_items = []
    neg_items = []
    for d in batch:
        u_id = d["user_ids"]
        pos_item = d["pos_item_id"]
        # "Unseen" = items the user has NEVER interacted with, excluding the
        # held-out positive itself so it can't also be drawn as a "negative".
        test_items.append(pos_item)
        interacted = interacted_by_user.get(u_id, set()) | {pos_item}
        candidates = np.setdiff1d(all_items, np.fromiter(interacted, dtype=np.int64))

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


