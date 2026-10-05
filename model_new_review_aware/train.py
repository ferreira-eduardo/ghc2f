import sys
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# permite rodar como script (python .../train.py) além de python -m
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_new_review_aware.data import BatchBuilder, TrainStats
from model_new_review_aware.load_data import load_all_beauty
from model_new_review_aware.collaborative_branch import CollaborativeBranch
from model_new_review_aware.fielder_match import FieldMatcher, FieldProjector
from model_new_review_aware.losses import LossWeights, ReviewAwareLoss
from model_new_review_aware.review_aware import ITEM_FIELDS, USER_FIELDS, ReviewAware, Gate
from model_new_review_aware.text_encoder import TextEncoder
from model_new_review_aware.threshold import mark_liked, user_thresholds


@dataclass
class Config:
    root: str = "data/all_beauty"
    cf_dim: int = 64
    txt_dim: int = 128
    rank: int = 8
    batch_size: int = 512
    lr: float = 1e-3
    epochs: int = 30
    patience: int = 3       # épocas sem melhora no nDCG@10 da validação
    eval_chunk: int = 128   # usuários por vez na avaliação
    seed: int = 42
    output: str = "review_aware.pt"


# ---------- Avaliação ----------

def evaluation_rows(split: pd.DataFrame, catalog: dict, history: dict) -> pd.DataFrame:
    """Interações avaliáveis: positivas, item conhecido no treino e não repetido do histórico."""
    rows = split[split["liked"] & split["item"].isin(catalog)]
    fresh = [t not in history.get(u, ()) for u, t in zip(rows["user"], rows["item"])]
    return rows[fresh]


@torch.no_grad()
def evaluate(score_fn: Callable[[list], torch.Tensor], rows: pd.DataFrame, catalog: dict,
             history: dict, chunk: int, k: int = 10) -> dict[str, float]:
    """Ranking sobre todo o catálogo do treino; o histórico do usuário é excluído.

    score_fn(users) -> (b, N) scores, com N = len(catalog), na ordem de catalog.
    Empates contam pela posição média, para não favorecer scores constantes.
    """
    ranks = []
    for start in range(0, len(rows), chunk):
        part = rows.iloc[start:start + chunk]
        users = part["user"].tolist()
        scores = score_fn(users).float()

        for r, u in enumerate(users):
            seen = [catalog[i] for i in history.get(u, ()) if i in catalog]
            scores[r, seen] = float("-inf")

        target = torch.tensor([catalog[t] for t in part["item"]], device=scores.device)
        t_score = scores.gather(1, target[:, None])
        higher = (scores > t_score).sum(1)
        ties = (scores == t_score).sum(1) - 1
        ranks.append((higher + ties / 2).cpu())

    ranks = torch.cat(ranks).numpy()  # 0 = primeiro lugar
    return {
        f"ndcg@{k}": float(np.where(ranks < k, 1 / np.log2(ranks + 2), 0).mean()),
        f"recall@{k}": float((ranks < k).mean()),
        "recall@50": float((ranks < 50).mean()),
    }


# ---------- Treino ----------

def train_epoch(model: ReviewAware, loss_fn: ReviewAwareLoss, loader: DataLoader,
                optimizer: torch.optim.Optimizer, device) -> dict[str, float]:
    model.train()
    totals, steps = defaultdict(float), 0
    for batch in loader:
        batch = batch.to(device)
        scores = model(batch, in_batch=True)
        loss, terms = loss_fn(
            scores, ratings=batch.ratings, positive=batch.liked, item_log_q=batch.item_log_q,
            users=batch.users, items=batch.items, regularization=model.regularization(batch),
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        for name, value in terms.items():
            totals[name] += value
        steps += 1
    return {name: value / steps for name, value in totals.items()}


def trainable_state(model: ReviewAware, loss_fn: ReviewAwareLoss) -> dict:
    """Pesos treináveis em CPU; o encoder congelado fica de fora."""
    model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                   if not k.startswith("encoder.")}
    loss_state = {k: v.detach().cpu().clone() for k, v in loss_fn.state_dict().items()}
    return {"model": model_state, "loss": loss_state}


def main(cfg: Config = Config()) -> None:
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dados
    train, val, _, user_profiles, item_profiles = load_all_beauty(cfg.root)
    thresholds, fallback = user_thresholds(train)
    train = mark_liked(train, thresholds, fallback)
    val = mark_liked(val, thresholds, fallback)

    stats = TrainStats.from_train(train, user_profiles, item_profiles)
    builder = BatchBuilder(stats, user_profiles, item_profiles)
    loader = DataLoader(train[["user", "item", "rating", "liked"]].to_dict("records"),
                        batch_size=cfg.batch_size, shuffle=True, collate_fn=builder)

    catalog = {item: j for j, item in enumerate(stats.item_index)}  # itens do treino
    items = list(catalog)
    history = train.groupby("user")["item"].agg(set).to_dict()
    rows = evaluation_rows(val, catalog, history)
    print(f"validação: {len(rows)} de {len(val)} interações avaliáveis ({len(rows) / len(val):.1%})")

    # Referência: popularidade no treino
    pop = torch.tensor([stats.item_count[i] for i in items], dtype=torch.float32)
    pop_metrics = evaluate(lambda users: pop.expand(len(users), -1).clone(),
                           rows, catalog, history, cfg.eval_chunk)
    print("popularidade:", {k: round(v, 4) for k, v in pop_metrics.items()})

    # Modelo
    encoder = TextEncoder().to(device)
    texts = (t for p in (*user_profiles.values(), *item_profiles.values()) for t in p.values() if t.strip())
    start = time.time()
    encoder.warm_up(texts)
    print(f"cache do encoder: {len(encoder.cache)} textos em {time.time() - start:.0f}s")

    model = ReviewAware(
        collaborative=CollaborativeBranch(len(stats.user_index), len(stats.item_index), cfg.cf_dim),
        text_encoder=encoder,
        projector=FieldProjector(USER_FIELDS + ITEM_FIELDS, encoder.dim, cfg.txt_dim, cfg.rank),
        matcher=FieldMatcher(len(USER_FIELDS), len(ITEM_FIELDS)),
        gate=Gate(4),
    ).to(device)
    loss_fn = ReviewAwareLoss(LossWeights()).to(device)
    params = [p for p in (*model.parameters(), *loss_fn.parameters()) if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=cfg.lr)

    def model_scores(users: list) -> torch.Tensor:
        return model(builder.build(users, items).to(device), in_batch=True).total

    # Laço com early stopping no nDCG@10 da validação
    best, best_state, waited = -1.0, None, 0
    for epoch in range(1, cfg.epochs + 1):
        start = time.time()
        terms = train_epoch(model, loss_fn, loader, optimizer, device)
        model.eval()
        metrics = evaluate(model_scores, rows, catalog, history, cfg.eval_chunk)

        print(f"\népoca {epoch} ({time.time() - start:.0f}s)")
        print("  perda:", {k: round(v, 4) for k, v in terms.items()})
        print("  val:  ", {k: round(v, 4) for k, v in metrics.items()})

        if metrics["ndcg@10"] > best:
            best, best_state, waited = metrics["ndcg@10"], trainable_state(model, loss_fn), 0
        else:
            waited += 1
            if waited >= cfg.patience:
                print(f"\nparada antecipada: {cfg.patience} épocas sem melhora")
                break

    torch.save({"state": best_state, "config": cfg}, cfg.output)
    print(f"\nmelhor nDCG@10: {best:.4f} (popularidade: {pop_metrics['ndcg@10']:.4f})")

    # Diagnóstico do melhor modelo
    model.load_state_dict(best_state["model"], strict=False)
    grid = pd.DataFrame(model.matcher.weights.detach().cpu().numpy(),
                        index=USER_FIELDS, columns=ITEM_FIELDS)
    print("\npesos do casamento campo a campo:\n", grid.round(3))
    print("\nespecialização das projeções:", {k: round(v, 3) for k, v in model.projector.specialization().items()})
    print("pesos do gate:", model.gate.linear.weight.detach().cpu().numpy().round(3))


if __name__ == "__main__":
    main()