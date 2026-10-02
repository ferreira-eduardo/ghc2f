from dataclasses import dataclass, asdict

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


def false_negative_mask(users: Tensor, items: Tensor) -> Tensor:
    """
    (B, B): True onde a coluna é o mesmo item ou um item do mesmo usuário (inclui a diagonal).
    """
    return (items[None, :] == items[:, None]) | (users[None, :] == users[:, None])


def disliked_pairwise(logits: Tensor, positive: Tensor, users: Tensor, items: Tensor) -> Tensor:
    """
    L_pair: nas linhas com rating baixo, não observado > não gostou (BPR).
    A diagonal é o item rejeitado; as demais colunas válidas são itens não observados.
    """
    valid = ~false_negative_mask(users, items) & ~positive[:, None]
    if not valid.any():
        return logits.new_zeros(())

    # s(u, não observado) - s(u, não gostou)
    diff = logits - logits.diagonal()[:, None]
    return -F.logsigmoid(diff[valid]).mean()


@dataclass
class LossWeights:
    """
    Pontos de partida, a ajustar na validação.
    Peso zero desliga o termo.
    """
    sm: float = 1.0  # L_sm(s), fixo como referência
    pair: float = 0.1  # alpha
    ord: float = 0.1  # eta
    sm_cf: float = 0.5  # mu_1
    sm_txt: float = 0.5  # mu_2
    decorr: float = 0.1  # lambda
    cf_l2: float = 1e-4
    matcher_l1: float = 1e-4


class OrdinalLoss(nn.Module):
    """
    L_ord: P(r <= k) = sigmoid(theta_k - s),
    limiares aprendidos e ordenados.
    """

    def __init__(self, n_levels: int = 5):
        super().__init__()
        # theta_1 livre; os demais = theta_1 + soma de incrementos positivos (garante a ordem).
        self.first = nn.Parameter(torch.tensor(-1.5))
        self.gaps = nn.Parameter(torch.zeros(n_levels - 2))

    def thresholds(self) -> Tensor:
        steps = torch.cumsum(F.softplus(self.gaps), dim=0)
        return torch.cat([self.first.view(1), self.first + steps])

    def forward(self, scores: Tensor, ratings: Tensor) -> Tensor:
        """
        scores (B,), ratings (B,) inteiros de 1 a n_levels.
        """
        cdf = torch.sigmoid(self.thresholds()[None, :] - scores[:, None])  # (B, n-1)
        zeros, ones = cdf.new_zeros(len(cdf), 1), cdf.new_ones(len(cdf), 1)
        cdf = torch.cat([zeros, cdf, ones], dim=1)  # (B, n+1)
        probs = cdf[:, 1:] - cdf[:, :-1]  # P(r = k)
        p = probs.gather(1, (ratings.long() - 1)[:, None]).squeeze(1)
        return -torch.log(p.clamp(min=1e-9)).mean()


def decorrelation(a: Tensor, b: Tensor, eps: float = 1e-8) -> Tensor:
    """R_dec: quadrado da correlação de Pearson entre os scores dos dois ramos."""
    a = a.flatten() - a.mean()
    b = b.flatten() - b.mean()
    return ((a * b).sum() / (a.norm() * b.norm() + eps)) ** 2


def sampled_softmax(logits: Tensor, item_log_q: Tensor, positive: Tensor,
                    users: Tensor, items: Tensor) -> Tensor:
    """
    L_sm: classificação do item de cada linha contra os itens do batch.
    logits:     (B, B) scores do modo in-batch; linha = usuário, coluna = item.
    item_log_q: (B,) log da probabilidade de cada item do batch ter sido amostrado.
    positive:   (B,) True se a interação da linha é positiva (rating >= limiar).
    users, items: (B,) IDs, usados para mascarar falsos negativos.
    """
    # Correção logQ: itens populares aparecem mais como negativos; descontamos isso.
    logits = logits - item_log_q[None, :]

    # Falsos negativos: fora da diagonal, o mesmo item ou um item do mesmo usuário.
    eye = torch.eye(len(items), dtype=torch.bool, device=logits.device)
    same = (items[None, :] == items[:, None]) | (users[None, :] == users[:, None])
    logits = logits.masked_fill(same & ~eye, float("-inf"))

    # O alvo de cada linha é a diagonal.
    targets = torch.arange(len(items), device=logits.device)
    loss = F.cross_entropy(logits, targets, reduction="none")

    # Só linhas positivas contam; linhas com rating baixo ficam para L_pair e L_ord.
    return loss[positive].sum() / positive.sum().clamp(min=1)


class ReviewAwareLoss(nn.Module):
    """L = L_sm(s) + α·L_pair + η·L_ord + μ₁·L_sm(s_cf) + μ₂·L_sm(s_txt) + λ·R_dec + regularização."""

    def __init__(self, weights: LossWeights, n_levels: int = 5):
        super().__init__()
        self.weights = asdict(weights)
        self.ordinal = OrdinalLoss(n_levels)

    def forward(
            self,
            scores,
            ratings: Tensor,
            positive: Tensor,
            item_log_q: Tensor,
            users: Tensor,
            items: Tensor,
            regularization: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        """scores: saída do ReviewAware com in_batch=True (matrizes B x B)."""

        def rank(logits: Tensor) -> Tensor:
            return sampled_softmax(logits, item_log_q, positive, users, items)

        terms = {
            "sm": rank(scores.total),
            "pair": disliked_pairwise(scores.total, positive, users, items),
            "ord": self.ordinal(scores.total.diagonal(), ratings),
            "sm_cf": rank(scores.collaborative),
            "sm_txt": rank(scores.textual),
            "decorr": decorrelation(scores.collaborative, scores.textual),
            **regularization,
        }

        total = sum(self.weights[name] * value for name, value in terms.items())

        return total, {name: value.item() for name, value in terms.items()}


if __name__ == "__main__":
    loss_fn = ReviewAwareLoss(LossWeights())
    print(loss_fn.ordinal.thresholds())  # crescente: tensor([-1.5, -0.81, -0.11, 0.58])

    # L_pair: linha 1 com rating baixo; itens não observados acima do rejeitado -> perda baixa
    logits = torch.tensor([[5.0, 1.0, 1.0], [3.0, -2.0, 3.0], [1.0, 1.0, 5.0]])
    ids = torch.tensor([1, 2, 3])
    print(disliked_pairwise(logits, torch.tensor([True, False, True]), ids, ids))  # ~0.0067

    # R_dec: scores idênticos -> 1; independentes -> próximo de 0
    a = torch.randn(100)
    print(decorrelation(a, a), decorrelation(a, torch.randn(100)))

    # Ordinal: score alto deve favorecer nota alta
    print(loss_fn.ordinal(torch.tensor([3.0]), torch.tensor([5])),
          loss_fn.ordinal(torch.tensor([3.0]), torch.tensor([1])))
