from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


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




class AspectDataset(Dataset):
    def __init__(self, user_item_matrix, df_full, all_review_embeddings):
        self.matrix = user_item_matrix
        self.review_embeddings = torch.tensor(all_review_embeddings, dtype=torch.float32)

        # Agrupamos os índices das reviews por usuário
        self.user_reviews_idx = df_full.groupby('userId').apply(lambda x: x.index.tolist()).to_dict()

        # Mapeamento (User, Item) -> Index da review específica
        self.interaction_to_idx = {(r.userId, r.itemId): i for i, r in enumerate(df_full.itertuples())}

        self.user_interactions = df_full.groupby('userId')['itemId'].apply(list).to_dict()
        self.item_repr = df_full.groupby('itemId').apply(lambda x: x.index.tolist()).to_dict()

    def __getitem__(self, u_idx):
        # 1. Recuperar Histórico de Texto do Usuário
        # Todas as reviews que este usuário já fez
        h_indices = self.user_reviews_idx.get(u_idx, [0])
        user_history_emb = self.review_embeddings[h_indices].mean(dim=0, keepdim=True)  # [1, 768]

        # 2. Item Positivo e sua Review específica
        pos_items = self.user_interactions.get(u_idx, [0])
        pos_item = np.random.choice(pos_items)
        pos_rev_idx = self.interaction_to_idx[(u_idx, pos_item)]
        pos_text = self.review_embeddings[pos_rev_idx].unsqueeze(0)

        # 3. Item Negativo e uma Review "exemplo" dele
        neg_item = np.random.choice(np.arange(self.matrix.shape[1]))
        while neg_item in self.user_interactions.get(u_idx, []):
            neg_item = np.random.choice(np.arange(self.matrix.shape[1]))

        neg_rev_indices = self.item_repr.get(neg_item, [0])
        neg_text = self.review_embeddings[np.random.choice(neg_rev_indices)].unsqueeze(0)

        return {
            "user_ids": u_idx,
            "pos_item_id": pos_item,
            "neg_item_id": neg_item,
            "ratings_in": torch.from_numpy(self.matrix[u_idx].toarray()).float().squeeze(),
            "user_history_text": user_history_emb,
            "pos_text_seq": pos_text,
            "neg_text_seq": neg_text
        }