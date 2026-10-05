from dataclasses import dataclass, replace, fields

import torch
import torch.nn as nn
from torch import Tensor

from model_new_review_aware.collaborative_branch import CollaborativeBranch
from model_new_review_aware.fielder_match import FieldMatcher, FieldProjector
from model_new_review_aware.text_encoder import TextEncoder

USER_FIELDS = ("likes", "dislikes", "context")
ITEM_FIELDS = ("summary", "strengths", "weaknesses", "suits")


@dataclass
class Batch:
    users: Tensor  # (B,) índices; 0 = desconhecido no treino
    items: Tensor  # (B,)
    ratings: Tensor  # (B,) notas brutas (perda ordinal)
    liked: Tensor  # (B,) bool, limiar por usuário
    item_log_q: Tensor  # (B,) log da prob. de amostragem (correção logQ)
    user_texts: dict[str, list[str]]  # campo -> B textos ("" quando ausente)
    item_texts: dict[str, list[str]]
    user_features: Tensor  # (B, f_u) sinais de confiabilidade do usuário
    item_features: Tensor  # (B, f_i) sinais de confiabilidade do item

    def to(self, device) -> "Batch":
        """Move os tensores; os textos ficam como estão."""
        moved = {f.name: getattr(self, f.name).to(device) for f in fields(self)
                 if isinstance(getattr(self, f.name), Tensor)}
        return replace(self, **moved)



@dataclass
class Scores:
    total: Tensor
    collaborative: Tensor
    textual: Tensor
    gate: Tensor


@dataclass
class LossWeights:
    """Pontos de partida, a ajustar na validação. Peso zero desliga o termo."""
    sm: float = 1.0           # L_sm(s), fixo como referência
    pair: float = 0.1         # alpha
    ord: float = 0.1          # eta
    sm_cf: float = 0.5        # mu_1
    sm_txt: float = 0.5       # mu_2
    decorr: float = 0.1       # lambda
    cf_l2: float = 1e-4
    matcher_l1: float = 1e-4



class Gate(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, features: Tensor) -> Tensor:
        return torch.sigmoid(self.linear(features)).squeeze(-1)


def pair(user_side: Tensor, item_side: Tensor, in_batch: bool) -> tuple[Tensor, Tensor]:
    """Alinha os lados: pares (B,) x (B,) ou todos contra todos (B, 1) x (1, B)."""
    if in_batch:
        return user_side[:, None], item_side[None]
    return user_side, item_side

class ReviewAware(nn.Module):
    """
    s = s_cf + g * s_txt
    """
    def __init__(
            self,
            collaborative: CollaborativeBranch | None,
            text_encoder: TextEncoder | None,
            projector: FieldProjector | None,
            matcher: FieldMatcher | None,
            gate: Gate | None,
    ):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        super().__init__()
        self.collaborative = collaborative
        self.text_encoder = text_encoder
        self.projector = projector
        self.matcher = matcher
        self.gate = gate

    def encode_fields(self, texts: dict[str, list[str]],
                      fields: tuple[str, ...]) -> tuple[Tensor, Tensor]:
        """
        Textos por campo -> vetores projetados (B, F, d) e máscara de presença (B, F).
        """
        rows = list(zip(*(texts[f] for f in fields)))
        rows = [[t.strip() for t in row] for row in rows]

        device = self.text_encoder.device
        mask = torch.tensor([[bool(t) for t in row] for row in rows], device=device)
        embs = torch.zeros(len(rows), len(fields), self.text_encoder.dim, device=device)
        present = [t for row in rows for t in row if t]
        if present:
            embs[mask] = self.text_encoder(present)

        return self.projector(embs, fields), mask

    def forward(self, batch: Batch, in_batch: bool = False) -> Scores:
        """
        in_batch=False: score de cada par (B,).
         in_batch=True: todos contra todos (B, B).
         """
        users, items = pair(batch.users, batch.items, in_batch)
        zeros = torch.zeros(torch.broadcast_shapes(users.shape, items.shape), device=users.device)

        s_cf = self.collaborative(users, items) if self.collaborative is not None else zeros
        s_txt = self._text_score(batch, in_batch) if self.text_encoder is not None else zeros
        g = self._gate(batch, in_batch) if self.gate is not None else torch.ones_like(zeros)

        return Scores(total=s_cf + g * s_txt, collaborative=s_cf, textual=s_txt, gate=g)

    def _text_score(self, batch: Batch, in_batch: bool) -> Tensor:
        user_vecs, user_mask = self.encode_fields(batch.user_texts, USER_FIELDS)
        item_vecs, item_mask = self.encode_fields(batch.item_texts, ITEM_FIELDS)
        user_vecs, item_vecs = pair(user_vecs, item_vecs, in_batch)
        user_mask, item_mask = pair(user_mask, item_mask, in_batch)
        return self.matcher(user_vecs, item_vecs, user_mask, item_mask)

    def _gate(self, batch: Batch, in_batch: bool) -> Tensor:
        u, i = pair(batch.user_features, batch.item_features, in_batch)
        shape = torch.broadcast_shapes(u.shape[:-1], i.shape[:-1])
        x = torch.cat([u.expand(*shape, -1), i.expand(*shape, -1)], dim=-1)
        return self.gate(x)

    def regularization(self, batch: Batch) -> dict[str, Tensor]:
        """
        Termos de regularização, ponderados na perda (fora do modelo).
        """
        terms = {}
        if self.collaborative is not None:
            terms["cf_l2"] = self.collaborative.regularization(batch.users, batch.items)
        if self.matcher is not None:
            terms["matcher_l1"] = self.matcher.regularization()
        return terms



if __name__ == "__main__":
    # B = 3
    # batch = Batch(
    #     users=torch.tensor([1, 2, 3]), items=torch.tensor([4, 5, 1]),
    #     user_texts={"likes": ["warm tone", "light weight", "bright sound"],
    #                 "dislikes": ["", "buzzing frets", ""],
    #                 "context": ["home practice", "gigs", "recording"]},
    #     item_texts={"summary": ["vintage guitar", "travel guitar", "studio guitar"],
    #                 "strengths": ["warm tone", "light", "clear sound"],
    #                 "weaknesses": ["heavy", "", "expensive"],
    #                 "suits": ["blues players", "travelers", "studios"]},
    #     user_features=torch.randn(B, 2), item_features=torch.randn(B, 2),
    # )
    #
    # model = ReviewAware(CollaborativeBranch(5, 5, 64), TextEncoder(),
    #                     FieldProjector(USER_FIELDS + ITEM_FIELDS, 768, 64, 4),
    #                     FieldMatcher(len(USER_FIELDS), len(ITEM_FIELDS)), Gate(4))
    #
    # print(model(batch).total.shape)                  # (3,)
    # print(model(batch, in_batch=True).total.shape)

    logits = torch.tensor([[5.0, 1.0, 1.0],
                           [1.0, 5.0, 1.0],
                           [1.0, 1.0, 5.0]])
    log_q = torch.zeros(3)
    users, items = torch.tensor([1, 2, 3]), torch.tensor([10, 20, 30])

    print(sampled_softmax(logits, log_q, torch.tensor([True, True, True]), users, items))  # baixo: diagonal dominante
    print(sampled_softmax(logits.T.flip(0), log_q, torch.ones(3, dtype=torch.bool), users, items))  # mais alto
    print(sampled_softmax(logits, log_q, torch.tensor([True, False, False]), users, items))  # só a linha 0 conta

    # Falso negativo: usuário 1 nas linhas 0 e 1 -> a coluna 1 some da linha 0
    print(sampled_softmax(logits, log_q, torch.ones(3, dtype=torch.bool), torch.tensor([1, 1, 3]), items))