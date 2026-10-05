import time
from collections import defaultdict
from collections.abc import Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from model_new_review_aware.load_data import load_all_beauty
from model_new_review_aware.collaborative_branch import CollaborativeBranch
from model_new_review_aware.data import BatchBuilder, TrainStats
from model_new_review_aware.fielder_match import FieldMatcher, FieldProjector
from model_new_review_aware.review_aware import Gate, ITEM_FIELDS, USER_FIELDS, ReviewAware
from model_new_review_aware.hyperparameters_search.hyper_params import HParams
from model_new_review_aware.losses import ReviewAwareLoss
from model_new_review_aware.text_encoder import TextEncoder
from model_new_review_aware.threshold import mark_liked, user_thresholds


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


def build_model(hp: HParams, stats: TrainStats, encoder: TextEncoder | None) -> ReviewAware:
    """Monta a variante pedida; partes ausentes ficam como None."""
    return ReviewAware(
        collaborative=CollaborativeBranch(len(stats.user_index), len(stats.item_index), hp.cf_dim)
        if hp.uses_cf else None,
        text_encoder=encoder if hp.uses_text else None,
        projector=FieldProjector(USER_FIELDS + ITEM_FIELDS, encoder.dim, hp.txt_dim, hp.rank)
        if hp.uses_text else None,
        matcher=FieldMatcher(len(USER_FIELDS), len(ITEM_FIELDS)) if hp.uses_text else None,
        gate=Gate(4) if hp.uses_gate else None,
    )


def main(hp: HParams = HParams(), root: str = "data/all_beauty",
         output: str | None = "review_aware.pt", eval_chunk: int = 128,
         encoder: TextEncoder | None = None,
         on_epoch: Callable[[int, dict], None] | None = None) -> dict[str, float]:
    """Treina uma configuração e devolve as métricas de validação do melhor estado.

    encoder:  um TextEncoder já aquecido, para reaproveitar o cache entre execuções.
    on_epoch: chamado com (época, métricas) após cada avaliação; o Optuna o usa para podar.
    """
    torch.manual_seed(hp.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("hiperparâmetros:", hp.to_dict())

    # Dados
    train, val, _, user_profiles, item_profiles = load_all_beauty(root)
    thresholds, fallback = user_thresholds(train, k=hp.shrink_k)
    train = mark_liked(train, thresholds, fallback)
    val = mark_liked(val, thresholds, fallback)

    stats = TrainStats.from_train(train, user_profiles, item_profiles)
    builder = BatchBuilder(stats, user_profiles, item_profiles)
    loader = DataLoader(train[["user", "item", "rating", "liked"]].to_dict("records"),
                        batch_size=hp.batch_size, shuffle=True, collate_fn=builder)

    catalog = {item: j for j, item in enumerate(stats.item_index)}  # itens do treino
    items = list(catalog)
    history = train.groupby("user")["item"].agg(set).to_dict()
    rows = evaluation_rows(val, catalog, history)
    print(f"validação: {len(rows)} de {len(val)} interações avaliáveis ({len(rows) / len(val):.1%})")

    # Referência: popularidade no treino
    pop = torch.tensor([stats.item_count[i] for i in items], dtype=torch.float32)
    pop_metrics = evaluate(lambda users: pop.expand(len(users), -1).clone(),
                           rows, catalog, history, eval_chunk)
    print("popularidade:", {k: round(v, 4) for k, v in pop_metrics.items()})

    # Encoder (só se a variante usa texto)
    if hp.uses_text and encoder is None:
        encoder = TextEncoder().to(device)
        texts = (t for p in (*user_profiles.values(), *item_profiles.values())
                 for t in p.values() if t.strip())
        start = time.time()
        encoder.warm_up(texts)
        print(f"cache do encoder: {len(encoder.cache)} textos em {time.time() - start:.0f}s")

    model = build_model(hp, stats, encoder).to(device)
    loss_fn = ReviewAwareLoss(hp.loss_weights()).to(device)
    params = [p for p in (*model.parameters(), *loss_fn.parameters()) if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=hp.lr)

    def model_scores(users: list) -> torch.Tensor:
        return model(builder.build(users, items).to(device), in_batch=True).total

    # Laço com early stopping no nDCG@10 da validação
    best, best_metrics, best_state, waited = -1.0, {}, None, 0
    for epoch in range(1, hp.epochs + 1):
        start = time.time()
        terms = train_epoch(model, loss_fn, loader, optimizer, device)
        model.eval()
        metrics = evaluate(model_scores, rows, catalog, history, eval_chunk)

        print(f"\népoca {epoch} ({time.time() - start:.0f}s)")
        print("  perda:", {k: round(v, 4) for k, v in terms.items()})
        print("  val:  ", {k: round(v, 4) for k, v in metrics.items()})
        if on_epoch is not None:
            on_epoch(epoch, metrics)

        if metrics["ndcg@10"] > best:
            best, best_metrics, waited = metrics["ndcg@10"], metrics, 0
            best_state = trainable_state(model, loss_fn)
        else:
            waited += 1
            if waited >= hp.patience:
                print(f"\nparada antecipada: {hp.patience} épocas sem melhora")
                break

    if output:
        torch.save({"state": best_state, "hparams": hp.to_dict()}, output)
    print(f"\nmelhor nDCG@10: {best:.4f} (popularidade: {pop_metrics['ndcg@10']:.4f})")

    # Diagnóstico do melhor modelo
    model.load_state_dict(best_state["model"], strict=False)
    if model.matcher is not None:
        grid = pd.DataFrame(model.matcher.weights.detach().cpu().numpy(),
                            index=USER_FIELDS, columns=ITEM_FIELDS)
        print("\npesos do casamento campo a campo:\n", grid.round(3))
        print("\nespecialização das projeções:",
              {k: round(v, 3) for k, v in model.projector.specialization().items()})
    if model.gate is not None:
        print("pesos do gate:", model.gate.linear.weight.detach().cpu().numpy().round(3))

    return best_metrics


if __name__ == "__main__":
    main()