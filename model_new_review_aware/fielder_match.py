import torch
import torch.nn.functional as F
from torch import Tensor, nn


class CellStandardizer(nn.Module):
    """
    Padroniza cada célula (p, q) com média e variância acumuladas.
    usa sempre as estatísticas acumuladas, em treino e em avaliação, para que
    o score de um par não dependa dos outros pares do batch.
    """

    def __init__(self, shape: tuple[int, int], momentum: float = 0.01, eps: float = 1e-5):
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.register_buffer("mean", torch.zeros(shape))
        self.register_buffer("var", torch.ones(shape))

    def forward(self, z: Tensor, mask: Tensor) -> Tensor:
        if self.training:
            self._update(z.detach(), mask)
        return (z - self.mean) / (self.var + self.eps).sqrt()

    @torch.no_grad()
    def _update(self, z: Tensor, mask: Tensor) -> None:
        dims = tuple(range(z.dim() - 2))  # todas as dimensões, exceto (U, I)
        m = mask.to(z.dtype)
        count = m.sum(dims)
        present = count > 0
        count = count.clamp(min=1)

        mean = (z * m).sum(dims) / count
        var = ((z - mean) ** 2 * m).sum(dims) / count

        self.mean.copy_(torch.where(present, torch.lerp(self.mean, mean, self.momentum), self.mean))
        self.var.copy_(torch.where(present, torch.lerp(self.var, var, self.momentum), self.var))


class FieldMatcher(nn.Module):
    """s_txt = sum_pq w_pq * mask_pq * z_pq, com z padronizado por célula.

    Formas broadcastáveis, como no CollaborativeBranch:
    user_vecs (B, U, d) e item_vecs (B, I, d) -> (B,);
    user_vecs (B, 1, U, d) e item_vecs (B, K, I, d) -> (B, K).
    """

    def __init__(self, n_user_fields: int, n_item_fields: int, momentum: float = 0.01):
        super().__init__()
        # Zero: o modelo começa como CF puro e o texto precisa conquistar seu peso.
        self.weights = nn.Parameter(torch.zeros(n_user_fields, n_item_fields))
        self.standardizer = CellStandardizer((n_user_fields, n_item_fields), momentum)

    def similarities(self, user_vecs: Tensor, item_vecs: Tensor) -> Tensor:
        """
        Cosseno entre usuário x item: (..., U, I).
        """
        u = F.normalize(user_vecs, dim=-1)
        i = F.normalize(item_vecs, dim=-1)
        return u @ i.transpose(-1, -2)

    def contributions(self, user_vecs: Tensor, item_vecs: Tensor,
                      user_mask: Tensor, item_mask: Tensor) -> Tensor:
        """Contribuição de cada célula ao score: (..., U, I). Base da interpretação."""
        mask = user_mask.unsqueeze(-1) & item_mask.unsqueeze(-2)
        z = self.standardizer(self.similarities(user_vecs, item_vecs), mask)
        return torch.where(mask, self.weights * z, 0.0)

    def forward(self, user_vecs: Tensor, item_vecs: Tensor,
                user_mask: Tensor, item_mask: Tensor) -> Tensor:
        return self.contributions(user_vecs, item_vecs, user_mask, item_mask).sum(dim=(-2, -1))

    def regularization(self) -> Tensor:
        """L1 nos pesos: empurra células irrelevantes para zero."""
        return self.weights.abs().sum()


class FieldProjector(nn.Module):
    """
    Projeção para o espaço comum: P_f = P + B_f A_f.
    Um projetor para todos os campos (usuário e item)
    """

    def __init__(self, fields: tuple[str, ...], in_dim: int, out_dim: int, rank: int):
        super().__init__()
        self.shared = nn.Linear(in_dim, out_dim, bias=False)
        self.down = nn.ModuleDict({f: nn.Linear(in_dim, rank, bias=False) for f in fields})
        self.up = nn.ModuleDict({f: nn.Linear(rank, out_dim, bias=False) for f in fields})

        # Ortogonal: preserva aproximadamente a geometria do encoder ao reduzir a dimensão.
        nn.init.orthogonal_(self.shared.weight)
        # up = 0: o ajuste por campo começa nulo (down mantém a inicialização padrão).
        for f in fields:
            nn.init.zeros_(self.up[f].weight)

    def forward(self, embs: Tensor, fields: tuple[str, ...]) -> Tensor:
        """(..., F, in_dim) -> (..., F, out_dim), com F = len(fields), na mesma ordem."""
        deltas = [self.up[f](self.down[f](embs[..., k, :])) for k, f in enumerate(fields)]
        return self.shared(embs) + torch.stack(deltas, dim=-2)

    @torch.no_grad()
    def specialization(self) -> dict[str, float]:
        """Norma de B_f A_f por campo: quanto cada campo se afastou da projeção comum."""
        return {f: (self.up[f].weight @ self.down[f].weight).norm().item() for f in self.up}


if __name__ == "__main__":
    """ Class test Fielder Matcher """
    #
    # fm = FieldMatcher(3, 4)
    # with torch.no_grad():
    #     fm.weights.fill_(1.0)
    # print(fm.weights.sum())
    #
    # uv, iv = torch.randn(2, 3, 8), torch.randn(2, 4, 8)
    # um = torch.tensor([[True, False, True]] * 2)
    # im = torch.ones(2, 4, dtype=torch.bool)
    #
    # c = fm.contributions(uv, iv, um, im)
    # print(c[:, 1].abs().sum())  # 0
    # print(c[:, 0].abs().sum())  # > 0

    """ Class test Fielder Projector"""
    USER_FIELDS = ("likes", "dislikes", "context")                      # 3 campos
    ITEM_FIELDS = ("summary", "strengths", "weaknesses", "suits")


    proj = FieldProjector(USER_FIELDS + ITEM_FIELDS, in_dim=16, out_dim=8, rank=2)
    embs = torch.randn(2, 3, 16)
    out = proj(embs, USER_FIELDS)
    print(out.shape)  # (2, 3, 8)
    print(torch.allclose(out, proj.shared(embs)))  # True: ajustes começam nulos
    print(proj.specialization())  # todos 0.0
    print(proj(torch.randn(2, 5, 4, 16), ITEM_FIELDS).shape)