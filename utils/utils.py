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


def prepare_inputs(df, entity_col, topic_col_names, max_reviews=50):
    vectors = df[topic_col_names].values

    # Create a temporary series to group
    temp_df = pd.DataFrame({
        entity_col: df[entity_col],
        'vec': list(vectors)
    })

    # Group by User/Item
    grouped = temp_df.groupby(entity_col)['vec'].apply(lambda x: x[:max_reviews]).reset_index()

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



class EarlyStoppingRanking:
    def __init__(self, patience=5, delta=0, verbose=True):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_state = None

    def __call__(self, current_metric, model):
        score = current_metric

        if self.best_score is None:
            self.best_score = score
            self.stash_best_state(model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.stash_best_state(model)
            self.counter = 0

    def stash_best_state(self, model):
        if self.verbose:
            print("New best metric...")
        self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    def load_best_into_model(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
            if self.verbose:
                print("Best state restored...")


def train_collate_fn_aspect(batch, text_embeddings, user_reviews_dict):
    user_ids = torch.tensor([x['user_id'] for x in batch])
    pos_item_ids = torch.tensor([x['pos_item_id'] for x in batch])
    neg_item_ids = torch.tensor([x['neg_item_id'] for x in batch])
    ratings_in = torch.stack([x['ratings_in'] for x in batch])

    pos_text_seq = torch.stack([x['pos_text_seq'] for x in batch])
    neg_text_seq = torch.stack([x['neg_text_seq'] for x in batch])

    # 2. Gerar histórico do usuário (Média das reviews passadas)
    user_histories = []
    for u in user_ids.tolist():
        h_idx = user_reviews_dict[u]
        # Pegamos a média dos embeddings das reviews que o usuário já fez
        u_emb = text_embeddings[h_idx].mean(dim=0, keepdim=True)
        user_histories.append(u_emb)

    user_history_text = torch.stack(user_histories)  # [B, 1, 768]

    return {
        "user_ids": user_ids,
        "pos_item_id": pos_item_ids,
        "neg_item_id": neg_item_ids,
        "ratings_in": ratings_in,
        "pos_text_seq": pos_text_seq,
        "neg_text_seq": neg_text_seq,
        "user_history_text": user_history_text
    }



class AspectDataset(Dataset):
    def __init__(self, user_item_matrix, df, all_review_embeddings):
        self.matrix = user_item_matrix
        self.df = df
        self.review_embeddings = torch.from_numpy(all_review_embeddings).float()
        # Agrupamos os índices das reviews por usuário
        unique_users = df['userId'].unique()
        self.user_to_matrix_idx = {user: i for i, user in enumerate(unique_users)}
        unique_items = df['itemId'].unique()
        self.item_to_matrix_idx = {item: i for i, item in enumerate(unique_items)}

        self.user_reviews_idx = self.df.groupby('userId').indices
        self.item_repr = {k: list(v) for k, v in self.df.groupby('itemId').indices.items()}
        self.user_interactions = self.df.groupby('userId')['itemId'].apply(set).to_dict()

        self.num_items = user_item_matrix.shape[1]
        self.all_items = np.arange(self.num_items)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        user_id = int(row['userId'])
        pos_item_id = int(row['itemId'])

        # 1. Sorteio do Negativo
        neg_item_id = np.random.choice(self.all_items)
        while neg_item_id in self.user_interactions.get(user_id, set()):
            neg_item_id = np.random.choice(self.all_items)

        # 2. Pegar os embeddings de texto
        # O positivo é a review da linha atual
        pos_text = self.review_embeddings[idx].unsqueeze(0)

        # O NEGATIVO: Como ele não tem uma "review" associada a essa interação,
        # pegamos uma review qualquer que exista para esse item negativo na base.
        # Se o item não tiver reviews, usamos um vetor de zeros ou a média global.
        neg_review_indices = self.item_repr.get(neg_item_id, [])
        if len(neg_review_indices) > 0:
            random_neg_idx = np.random.choice(neg_review_indices)
            neg_text = self.review_embeddings[random_neg_idx].unsqueeze(0)
        else:
            neg_text = torch.zeros((1, 768))  # Vetor nulo se o item for "frio"

        return {
            "user_id": user_id,
            "pos_item_id": pos_item_id,
            "neg_item_id": neg_item_id,
            "pos_text_seq": pos_text,
            "neg_text_seq": neg_text,  # <--- A CHAVE QUE ESTAVA FALTANDO
            "ratings_in": torch.from_numpy(self.matrix[user_id].toarray()).float().squeeze()
        }