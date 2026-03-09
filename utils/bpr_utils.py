import pandas as pd
import torch
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset


class BPRDataset(Dataset):
    def __init__(self, user_item_matrix, user_topics, num_negatives=1):
        self.ratings = user_item_matrix  # CSR Matrix
        self.user_topics = user_topics  # Tensor (num_users, topics_dim)
        self.num_users, self.num_items = user_item_matrix.shape
        self.num_negatives = num_negatives

        # Pre-calculate positive items for fast O(1) negative sampling
        self.user_to_pos = {}
        for u in range(self.num_users):
            # Efficiently get indices from CSR row
            pos_items = user_item_matrix[u].indices if hasattr(user_item_matrix, 'indices') else \
                np.where(user_item_matrix[u] > 0)[0]
            self.user_to_pos[u] = set(pos_items)

        # List of all (user, pos_item) pairs
        self.samples = [(u, i) for u, pos_set in self.user_to_pos.items() for i in pos_set]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        u, i = self.samples[idx]

        # Negative sampling: Find an item the user has NOT interacted with
        j = np.random.randint(0, self.num_items)
        while j in self.user_to_pos[u]:
            j = np.random.randint(0, self.num_items)

        # ratings_in: The full interaction vector for the AE Encoder
        # We convert the sparse row to a dense array only when needed
        row = self.ratings[u].toarray().squeeze() if hasattr(self.ratings, 'toarray') else self.ratings[u]

        return {
            "user_ids": torch.tensor(u, dtype=torch.long),
            "pos_item_ids": torch.tensor(i, dtype=torch.long),
            "neg_item_ids": torch.tensor(j, dtype=torch.long),
            "ratings_in": torch.FloatTensor(row),
            "user_topics": self.user_topics[u],
            "user_mask": (self.user_topics[u] != 0).float()
        }


def prepare_bpr_data(df, binarize=False):
    """
    df: DataFrame with columns ['userId', 'itemId', 'rating']
    """
    # 1. Extract values
    user_indices = df['userId'].values
    item_indices = df['itemId'].values

    if binarize:
        # For BPR, we usually treat any interaction as a 1
        values = np.ones(len(df))
    else:
        values = df['rating'].values

    # Create CSR Matrix
    user_item_matrix = csr_matrix(
        (values, (user_indices, item_indices)),
        shape=(df['userId'].nunique(), df['itemId'].nunique())
    )

    return user_item_matrix


def generate_user_topics(df, df_topics):
    """
    df: DataFrame with ['userId', 'itemId']
    df_topics: DataFrame where index is 'itemId' and columns are topic values
    """
    num_users = df.userId.nunique()
    topics_dim = df_topics.shape[1] - 2

    user_topics_matrix = np.zeros((num_users, topics_dim))

    # Group by user and calculate the mean of their topic interests
    user_profile_means = df_topics.groupby('userId').mean()

    # Populate the matrix
    for u_id in user_profile_means.index:
        if u_id < num_users:
            user_topics_matrix[u_id] = user_profile_means.loc[u_id].drop(['itemId'], errors='ignore').values

    return torch.FloatTensor(user_topics_matrix)


class EvalDataset(Dataset):
    def __init__(self, train_matrix, eval_df, user_topics):
        """
        train_matrix: The CSR matrix containing ONLY training data
        eval_df: The val_df or test_df (contains the one item per user)
        user_topics: The same user_topics_matrix used in training
        """
        self.train_matrix = train_matrix
        self.user_topics = user_topics

        # We only evaluate users who actually have an item in the eval set
        self.eval_data = eval_df[['userId', 'itemId']].values

    def __len__(self):
        return len(self.eval_data)

    def __getitem__(self, idx):
        u, target_item = self.eval_data[idx]
        u = int(u)

        row = self.train_matrix[u].toarray().squeeze()

        return {
            "user_ids": torch.tensor(u, dtype=torch.long),
            "target_item": torch.tensor(target_item, dtype=torch.long),
            "ratings_in": torch.FloatTensor(row),
            "user_topics": self.user_topics[u],
            "user_mask": (self.user_topics[u] != 0).float()
        }


class EarlyStoppingRanking:
    def __init__(self, patience=5, delta=0, verbose=True):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, current_metric, model, path):
        # current_metric should be something like HitRate@10
        score = current_metric

        if self.best_score is None:
            self.best_score = score
            # self.save_checkpoint(model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            # self.save_checkpoint(model, path)
            self.counter = 0

    def save_checkpoint(self, model, path):
        if self.verbose:
            print(f'Metric improved. Saving model to {path}...')
        torch.save(model.state_dict(), path)


