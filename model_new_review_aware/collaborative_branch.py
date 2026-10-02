import torch
from torch import Tensor, nn


class CollaborativeBranch(nn.Module):

    """
    BPR-MF de Rendle et al. (2009)
    MF com vieses treinada com perda de ranking
    s_cf = b_u + b_i + <p_u, q_i>.
    O índice 0 é reservado para IDs desconhecidos (não vistos no treino):
    vetor e viés fixos em zero, de modo que o ramo não contribui e o texto assume.
    Aceita formas broadcastáveis: users (B,) e items (B,) -> (B,);
    users (B, 1) e items (B, K) -> (B, K).
    """

    def __init__(self, n_users: int, n_items: int, dim: int, init_std: float = 0.01):
        super().__init__()
        self.user_embedding = nn.Embedding(n_users + 1, dim, padding_idx=0)
        self.item_embedding = nn.Embedding(n_items + 1, dim, padding_idx=0)
        self.user_bias = nn.Embedding(n_users + 1, 1, padding_idx=0)
        self.item_bias = nn.Embedding(n_items + 1, 1, padding_idx=0)
        self._init_weights(init_std)

    @torch.no_grad()
    def _init_weights(self, std: float) -> None:
        # O padrão do nn.Embedding é N(0, 1)
        nn.init.normal_(self.user_embedding.weight, std=std)
        nn.init.normal_(self.item_embedding.weight, std=std)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
        self.user_embedding.weight[0].zero_()
        self.item_embedding.weight[0].zero_()

    def forward(self, users: Tensor, items: Tensor) -> Tensor:
        u = self.user_embedding(users)
        i = self.item_embedding(items)
        bias = self.user_bias(users).squeeze(-1) + self.item_bias(items).squeeze(-1)
        return (u * i).sum(dim=-1) + bias

    def regularization(self, users: Tensor, items: Tensor) -> Tensor:
        """L2 apenas nas linhas usadas no batch."""
        p = self.user_embedding(users)
        q = self.item_embedding(items)
        return p.pow(2).sum(dim=-1).mean() + q.pow(2).sum(dim=-1).mean()