import numpy as np
import torch
from torch.utils.data import Dataset


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