from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


def _resolve_topic_cols(df: pd.DataFrame, topic_dim: int):
    str_cols = [str(i) for i in range(topic_dim)]
    if str_cols[0] in df.columns:
        return str_cols
    int_cols = list(range(topic_dim))
    if int_cols[0] in df.columns:
        return int_cols
    raise ValueError(
        f"Não encontrei colunas de tópicos como strings {str_cols[:3]}... "
        f"nem como ints {int_cols[:3]}... no df_topics."
    )

def build_user_item_topic_profiles(
    df_topics: pd.DataFrame,
    topic_dim: int,
    global_id_map: Dict[str, Dict[Any, int]],
    max_len: int = 50,
):
    """
    Espera que df_topics tenha pelo menos: userId, itemId, e colunas de tópicos.

    global_id_map pode ser:
      - {"user": {orig: idx}, "item": {orig: idx}}
        ou
      - {"user_map": {...}, "item_map": {...}}
    """
    # resolve maps
    user_map = global_id_map.get("user") or global_id_map.get("user_map")
    item_map = global_id_map.get("item") or global_id_map.get("item_map")
    if user_map is None or item_map is None:
        raise ValueError(
            "global_id_map deve conter {'user':..., 'item':...} "
            "ou {'user_map':..., 'item_map':...}."
        )

    topic_cols = _resolve_topic_cols(df_topics, topic_dim)

    user_topics, user_mask, user_idx = _build_one_profile(
        df_topics=df_topics,
        id_col="userId",
        topic_cols=topic_cols,
        topic_dim=topic_dim,
        id_map=user_map,
        max_len=max_len,
    )

    item_topics, item_mask, item_idx = _build_one_profile(
        df_topics=df_topics,
        id_col="itemId",
        topic_cols=topic_cols,
        topic_dim=topic_dim,
        id_map=item_map,
        max_len=max_len,
    )

    return {
        "user": (user_topics, user_idx, user_mask),
        "item": (item_topics, item_idx, item_mask),
    }


class AEDataset(Dataset):
    def __init__(self, rating_matrix: torch.Tensor, rating_y):
        # rating_matrix: [num_users, num_items]
        self.X = rating_matrix
        self.Y = rating_y

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        return self.X[idx]


def build_train_history(train_df_idx):
    history = {}
    grouped = train_df_idx.groupby('userId')
    for u_idx, group in grouped:
        items = group['itemId'].unique().tolist()
        history[u_idx] = set(items)
    return history


class HybridDataset(Dataset):
    def __init__(self, ratings_in, ratings_tgt, user_ids,
                 user_topics, user_mask):
        self.ratings_in = ratings_in
        self.ratings_tgt = ratings_tgt
        self.user_ids = user_ids
        self.user_topics = user_topics
        self.user_mask = user_mask

    def __len__(self):
        return self.ratings_in.shape[0]

    def __getitem__(self, u):
        return {
            "ratings_in": self.ratings_in[u],
            "ratings_tgt": self.ratings_tgt[u],
            "user_ids": self.user_ids[u],
            "user_topics": self.user_topics[u],
            "user_mask": self.user_mask[u],
        }


def build_global_id_maps(df_ratings: pd.DataFrame):
    user_ids = df_ratings["userId"].unique()
    item_ids = df_ratings["itemId"].unique()

    user_map = {uid: idx for idx, uid in enumerate(user_ids)}
    item_map = {iid: idx for idx, iid in enumerate(item_ids)}

    return {
        "user": user_map,
        "item": item_map
    }


def MSEloss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        size_average: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # mask of observed ratings
    mask = (targets != 0)
    mask_f = mask.float()

    # difference only on observed entries
    diff = (inputs - targets) * mask_f
    squared = diff.pow(2)

    # avoid division by zero
    num_ratings = mask_f.sum().clamp(min=1.0)

    if size_average:
        loss = squared.sum() / num_ratings
        norm = torch.tensor(1.0, device=loss.device)
    else:
        loss = squared.sum()
        norm = num_ratings

    return loss, norm


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



def _build_one_profile(
    df_topics: pd.DataFrame,
    id_col: str,
    topic_cols,
    topic_dim: int,
    id_map: Dict[Any, int],
    max_len: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Retorna:
      topics: (num_entities, max_len, topic_dim)
      mask:   (num_entities, max_len)  [float 0/1]
      idx:    (num_entities,)          [0..N-1]
    """
    num_entities = len(id_map)
    topics = torch.zeros((num_entities, max_len, topic_dim), dtype=torch.float32)
    mask = torch.zeros((num_entities, max_len), dtype=torch.float32)

    # filtra ids conhecidos + cria idx interno
    df_f = df_topics[df_topics[id_col].isin(id_map.keys())].copy()
    df_f["internal_idx"] = df_f[id_col].map(id_map)

    # se existir uma coluna de tempo, ordena p/ garantir "mais recentes"
    # (ajuste os nomes conforme o seu dataset)
    for tcol in ["timestamp", "unixReviewTime", "time", "date"]:
        if tcol in df_f.columns:
            df_f = df_f.sort_values(["internal_idx", tcol])
            break

    for idx_int, group in df_f.groupby("internal_idx", sort=False):
        raw = group[topic_cols].to_numpy()  # (n, topic_dim)

        # pega os últimos max_len (assumindo ordenado)
        if raw.shape[0] > max_len:
            raw = raw[-max_len:]

        seq_len = raw.shape[0]
        if seq_len == 0:
            continue

        topics[idx_int, :seq_len, :] = torch.as_tensor(raw, dtype=torch.float32)
        mask[idx_int, :seq_len] = 1.0

    entity_indices = torch.arange(num_entities, dtype=torch.long)
    return topics, mask, entity_indices


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
            self.save_checkpoint(model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model, path)
            self.counter = 0

    def save_checkpoint(self, model, path):
        if self.verbose:
            print(f'Metric improved. Saving model to {path}...')
        torch.save(model.state_dict(), path)