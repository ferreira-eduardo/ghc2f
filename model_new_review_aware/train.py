import sys
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# permite rodar como script (python .../train.py) além de python -m
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_new_review_aware.collaborative_branch import CollaborativeBranch
from model_new_review_aware.data import BatchBuilder, TrainStats
from model_new_review_aware.fielder_match import FieldMatcher, FieldProjector
from model_new_review_aware.hyperparameters_search.hyper_params import HParams
from model_new_review_aware.load_data import load_all_beauty
from model_new_review_aware.losses import ReviewAwareLoss
from model_new_review_aware.review_aware import ITEM_FIELDS, USER_FIELDS, Gate, ReviewAware
from model_new_review_aware.text_encoder import TextEncoder
from model_new_review_aware.threshold import mark_liked, rating_levels, user_thresholds

ITEM_CHUNK = 8192            # itens por vez na avaliação (limita a memória da GPU)
KS = (3, 5, 10, 15, 20)      # cortes reportados; o nDCG@10 é o critério de seleção
SELECTION = "ndcg@10"


# ---------- Ranking ----------

@dataclass
class Rankings:
    ranks: np.ndarray    # (n,) posição do alvo; 0 = primeiro lugar
    top: np.ndarray      # (n, max(KS)) índices do catálogo no topo de cada lista
    targets: np.ndarray  # (n,) índice do alvo no catálogo


def evaluation_rows(split: pd.DataFrame, catalog: dict, history: dict) -> pd.DataFrame:
    """Interações avaliáveis: positivas, item conhecido no treino e não repetido do histórico."""
    rows = split[split["liked"] & split["item"].isin(catalog)]
    fresh = [t not in history.get(u, ()) for u, t in zip(rows["user"], rows["item"])]
    return rows[fresh]


@torch.no_grad()
def rank_all(score_fn: Callable[[list], torch.Tensor], rows: pd.DataFrame, catalog: dict,
             history: dict, chunk: int) -> Rankings:
    """Uma passada sobre o catálogo do treino; todas as métricas saem daqui.

    score_fn(users) -> (b, N), na ordem de catalog. O histórico do usuário é excluído.
    Empates contam pela posição média, para não favorecer scores constantes.
    """
    ranks, tops, targets = [], [], []
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
        tops.append(scores.topk(max(KS), dim=1).indices.cpu())
        targets.append(target.cpu())

    return Rankings(torch.cat(ranks).numpy(), torch.cat(tops).numpy(), torch.cat(targets).numpy())


# ---------- Métricas ----------

def accuracy(ranks: np.ndarray, ks: tuple[int, ...] = KS) -> dict[str, float]:
    metrics = {}
    for k in ks:
        metrics[f"ndcg@{k}"] = float(np.where(ranks < k, 1 / np.log2(ranks + 2), 0).mean())
        metrics[f"recall@{k}"] = float((ranks < k).mean())
    return metrics


def head_items(popularity: np.ndarray, share: float = 0.2) -> np.ndarray:
    """Máscara da cabeça: os itens mais populares que somam `share` das interações."""
    order = np.argsort(-popularity)
    cumulative = np.cumsum(popularity[order]) / popularity.sum()
    head = np.zeros(len(popularity), dtype=bool)
    head[order[:np.searchsorted(cumulative, share) + 1]] = True
    return head


def beyond_accuracy(r: Rankings, popularity: np.ndarray, n_users: int, k: int = 10) -> dict[str, float]:
    """Cobertura, concentração e viés de popularidade das listas top-k."""
    top = r.top[:, :k]
    counts = np.bincount(top.ravel(), minlength=len(popularity)).astype(float)
    head = head_items(popularity)

    sorted_counts = np.sort(counts)
    i = np.arange(1, len(counts) + 1)
    gini = ((2 * i - len(counts) - 1) * sorted_counts).sum() / (len(counts) * counts.sum())

    p = counts[counts > 0] / counts.sum()
    entropy = -(p * np.log(p)).sum() / np.log(len(counts))  # normalizada em [0, 1]

    return {
        f"coverage@{k}": float((counts > 0).mean()),
        f"gini@{k}": float(gini),
        f"entropy@{k}": float(entropy),
        f"arp@{k}": float(popularity[top].mean()),
        f"aplt@{k}": float((~head[top]).mean()),
        f"novelty@{k}": float(-np.log2(popularity[top] / n_users).mean()),
    }


def stratified(r: Rankings, groups: np.ndarray, k: int = 10) -> dict[str, float]:
    """nDCG@k por grupo de usuários (ou de alvos)."""
    gains = np.where(r.ranks < k, 1 / np.log2(r.ranks + 2), 0)
    return {f"ndcg@{k}[{g}]": float(gains[groups == g].mean()) for g in np.unique(groups)}


def full_report(r: Rankings, popularity: np.ndarray, n_users: int,
                activity: np.ndarray) -> dict[str, float]:
    target_group = np.where(head_items(popularity)[r.targets], "alvo cabeça", "alvo cauda")
    return {**accuracy(r.ranks), **beyond_accuracy(r, popularity, n_users),
            **stratified(r, activity), **stratified(r, target_group)}


def activity_groups(rows: pd.DataFrame, user_count: dict) -> np.ndarray:
    """Faixas de interações no treino, onde se testa a hipótese do gate."""
    n = rows["user"].map(user_count).fillna(0).to_numpy()
    return np.select([n <= 1, n <= 3], ["1 interação", "2-3 interações"], "4+ interações")


@torch.no_grad()
def catalog_scorer(model: ReviewAware, builder: BatchBuilder, items: list, device) -> Callable[[list], torch.Tensor]:
    """score_fn(users) -> (b, N) contra todo o catálogo, para a avaliação.

    O lado dos itens (IDs, vetores de texto projetados, features do gate) não depende
    do usuário: é calculado uma vez aqui, com os pesos atuais, e reaproveitado em cada
    pedaço de usuários. Chamar de novo após mudar os pesos. O catálogo é pontuado em
    pedaços de ITEM_CHUNK itens para limitar a memória da GPU; cada par é pontuado de
    forma independente, então o resultado é o mesmo que de uma vez só.
    """
    catalog = model.item_side(builder.build([], items).to(device))

    def scores(users: list) -> torch.Tensor:
        u = model.user_side(builder.build(users, []).to(device))
        return torch.cat([model.score(u, catalog[j:j + ITEM_CHUNK], in_batch=True).total
                          for j in range(0, len(items), ITEM_CHUNK)], dim=1)

    return scores


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
                   if not k.startswith("text_encoder.")}
    loss_state = {k: v.detach().cpu().clone() for k, v in loss_fn.state_dict().items()}
    return {"model": model_state, "loss": loss_state}


def build_model(hp: HParams, stats: TrainStats, encoder: TextEncoder | None) -> ReviewAware:
    """Monta a variante pedida; partes ausentes ficam como None."""
    text = hp.uses_text
    return ReviewAware(
        collaborative=CollaborativeBranch(len(stats.user_index), len(stats.item_index), hp.cf_dim)
        if hp.uses_cf else None,
        text_encoder=encoder if text else None,
        projector=FieldProjector(USER_FIELDS + ITEM_FIELDS, encoder.dim, hp.txt_dim, hp.rank) if text else None,
        matcher=FieldMatcher(len(USER_FIELDS), len(ITEM_FIELDS)) if text else None,
        gate=Gate(4) if hp.uses_gate else None,
    )


def show(title: str, metrics: dict) -> None:
    print(f"  {title}:", {k: round(v, 4) for k, v in metrics.items()})


def main(hp: HParams = HParams(), root: str = "data/all_beauty",
         output: str | None = "review_aware.pt", eval_chunk: int = 128,
         encoder: TextEncoder | None = None,
         on_epoch: Callable[[int, dict], None] | None = None) -> dict[str, float]:
    """Treina uma configuração e devolve o relatório completo do melhor estado na validação.

    encoder:  TextEncoder já aquecido, para reaproveitar o cache entre execuções.
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
    popularity = np.array([stats.item_count[i] for i in items], dtype=float)
    n_users = train["user"].nunique()
    activity = activity_groups(rows, stats.user_count)
    print(f"validação: {len(rows)} de {len(val)} interações avaliáveis ({len(rows) / len(val):.1%})")

    # Referência: popularidade no treino
    pop = torch.tensor(popularity, dtype=torch.float32)
    pop_rankings = rank_all(lambda users: pop.expand(len(users), -1).clone(),
                            rows, catalog, history, eval_chunk)
    pop_report = full_report(pop_rankings, popularity, n_users, activity)

    # Encoder (só se a variante usa texto)
    if hp.uses_text and encoder is None:
        encoder = TextEncoder().to(device)
        texts = (t for p in (*user_profiles.values(), *item_profiles.values())
                 for t in p.values() if t.strip())
        start = time.time()
        encoder.warm_up(texts)
        print(f"cache do encoder: {len(encoder.cache)} textos em {time.time() - start:.0f}s")

    model = build_model(hp, stats, encoder).to(device)
    loss_fn = ReviewAwareLoss(hp.loss_weights(), n_levels=rating_levels(train)).to(device)
    params = [p for p in (*model.parameters(), *loss_fn.parameters()) if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=hp.lr)

    # Laço com early stopping no critério de seleção
    best, best_state, waited = -1.0, None, 0
    for epoch in range(1, hp.epochs + 1):
        start = time.time()
        terms = train_epoch(model, loss_fn, loader, optimizer, device)
        model.eval()
        metrics = accuracy(rank_all(catalog_scorer(model, builder, items, device), rows, catalog, history, eval_chunk).ranks)

        print(f"\népoca {epoch} ({time.time() - start:.0f}s)")
        show("perda", terms)
        show("val", {k: v for k, v in metrics.items() if k.endswith("@10")})
        if on_epoch is not None:
            on_epoch(epoch, metrics)

        if metrics[SELECTION] > best:
            best, best_state, waited = metrics[SELECTION], trainable_state(model, loss_fn), 0
        else:
            waited += 1
            if waited >= hp.patience:
                print(f"\nparada antecipada: {hp.patience} épocas sem melhora")
                break

    if output:
        torch.save({"state": best_state, "hparams": hp.to_dict()}, output)

    # Relatório completo do melhor estado, lado a lado com a popularidade
    model.load_state_dict(best_state["model"], strict=False)
    model.eval()
    report = full_report(rank_all(catalog_scorer(model, builder, items, device), rows, catalog, history, eval_chunk),
                         popularity, n_users, activity)
    table = pd.DataFrame({hp.variant: report, "popularidade": pop_report})
    print("\nrelatório na validação (melhor época):\n", table.round(4).to_string())

    # Diagnóstico interno
    if model.matcher is not None:
        grid = pd.DataFrame(model.matcher.weights.detach().cpu().numpy(),
                            index=USER_FIELDS, columns=ITEM_FIELDS)
        print("\npesos do casamento campo a campo:\n", grid.round(3))
        print("\nespecialização das projeções:",
              {k: round(v, 3) for k, v in model.projector.specialization().items()})
    if model.gate is not None:
        print("pesos do gate:", model.gate.linear.weight.detach().cpu().numpy().round(3))

    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--hparams", default=None, help="JSON exportado pelo tune.py")
    parser.add_argument("--seed", type=int, nargs="+", default=None,
                        help="várias sementes rodam em sequência")
    args = parser.parse_args()

    hp = HParams.load(args.hparams) if args.hparams else HParams()
    seeds = args.seed or [hp.seed]
    multi = len(seeds) > 1
    reports = {seed: main(replace(hp, seed=seed),
                          output=f"review_aware_seed{seed}.pt" if multi else "review_aware.pt")
               for seed in seeds}

    # Métricas por semente (colunas) e, com várias sementes, média e desvio
    Path("results").mkdir(exist_ok=True)
    prefix = f"results/review_aware_{hp.variant}"
    table = pd.DataFrame(reports)
    table.columns = [f"seed{s}" for s in table.columns]
    table.to_csv(f"{prefix}_seeds.csv", index_label="metric")
    print(f"\nmétricas salvas em {prefix}_seeds.csv")
    if multi:
        summary = pd.DataFrame({"média": table.mean(axis=1), "desvio": table.std(axis=1)})
        summary.to_csv(f"{prefix}_summary.csv", index_label="metric")
        print(f"resumo sobre as sementes {seeds} (salvo em {prefix}_summary.csv):\n",
              summary.round(4).to_string())