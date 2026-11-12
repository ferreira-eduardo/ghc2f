"""Evaluation protocol shared with the iRev baselines.

Ranking, metric and threshold code is copied from the baselines' evaluation module
so the GHC2F numbers are directly comparable. Only the Protocol constructor changes:
it takes the train / val / test dataframes (columns user, item, rating) instead of
reading iRev's .npy splits.

  - positive : rating >= user threshold (user_thresholds over train); used by
               recall, MRR and the beyond-accuracy metrics
  - nDCG     : every held-out row counts, gain = rating / max rating
  - catalog  : items seen in train; target must be in it and not in the history
  - ranking  : against the whole catalog, history excluded (train for val,
               train + val for test), ties count at their mean position
"""
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch


def rating_levels(train: pd.DataFrame) -> int:
    """Número de níveis da escala de notas (5 para 1-5, 10 para o IMDB 1-10), lido do treino.

    As notas precisam ser inteiras a partir de 1; a perda ordinal usa rating - 1 como índice.
    """
    r = train["rating"]
    if r.min() < 1 or (r % 1 != 0).any():
        raise ValueError(f"notas devem ser inteiras a partir de 1; encontrei min={r.min()}, "
                         f"valores {sorted(r.unique())[:10]}")
    return int(r.max())


def user_thresholds(train: pd.DataFrame, k: float = 5.0,
                    low: float = 0.6, high: float = 0.8) -> tuple[pd.Series, float]:
    """Limiar de positivo por usuário: média encolhida para a global, recortada em [low, high].

    low e high são frações da nota máxima: 0.6 e 0.8 dão [3, 4] na escala 1-5 e
    [6, 8] na escala 1-10. Calculado só com o treino. Devolve os limiares por usuário
    e o limiar global, usado para usuários que não aparecem no treino.
    """
    top = rating_levels(train)
    low, high = low * top, high * top
    global_mean = train["rating"].mean()
    stats = train.groupby("user")["rating"].agg(["sum", "count"])
    shrunk = (stats["sum"] + k * global_mean) / (stats["count"] + k)
    fallback = min(max(global_mean, low), high)
    return shrunk.clip(lower=low, upper=high), fallback


def mark_liked(df: pd.DataFrame, thresholds: pd.Series, fallback: float) -> pd.DataFrame:
    """Acrescenta a coluna 'liked'. Aplicar com os mesmos limiares em treino, validação e teste."""
    limit = df["user"].map(thresholds).fillna(fallback)
    return df.assign(liked=df["rating"] >= limit)


ITEM_CHUNK = 8192            # linhas avaliadas por vez (limita a memória da GPU)
KS = (3, 5, 10, 15, 20)      # cortes reportados; o nDCG@10 é o critério de seleção
SELECTION = "ndcg@10"


# ---------- Ranking ----------

@dataclass
class Rankings:
    ranks: np.ndarray    # (n,) posição do alvo; 0 = primeiro lugar
    top: np.ndarray      # (n, max(KS)) índices do catálogo no topo de cada lista
    targets: np.ndarray  # (n,) índice do alvo no catálogo
    ratings: np.ndarray  # (n,) nota original do alvo (relevância graduada do nDCG)
    liked: np.ndarray    # (n,) alvo positivo (nota >= limiar do usuário)

    def subset(self, mask: np.ndarray) -> "Rankings":
        return Rankings(self.ranks[mask], self.top[mask], self.targets[mask],
                        self.ratings[mask], self.liked[mask])


def evaluation_rows(split: pd.DataFrame, catalog: dict, history: dict) -> pd.DataFrame:
    """Interações avaliáveis: qualquer nota, item conhecido no treino e não repetido do histórico.

    Todas entram no nDCG (relevância = nota); recall, MRR e além da acurácia usam só as positivas.
    """
    rows = split[split["item"].isin(catalog)]
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

    return Rankings(torch.cat(ranks).numpy(), torch.cat(tops).numpy(), torch.cat(targets).numpy(),
                    rows["rating"].to_numpy(dtype=float), rows["liked"].to_numpy(dtype=bool))


# ---------- Métricas ----------

def graded_dcg(r: Rankings, top_rating: int, k: int) -> np.ndarray:
    """nDCG@k graduado por linha: relevância = nota / nota máxima da escala.

    Cada usuário tem um único alvo, então o DCG ideal é o de um item com a nota máxima
    no primeiro lugar (= 1). Assim um alvo nota 5 no topo vale 1 e um nota 1 vale 0.2
    (escala 1-5); com o IDCG do próprio alvo a nota se cancelaria e o nDCG voltaria a ser binário.
    """
    gain = r.ratings / top_rating
    return np.where(r.ranks < k, gain / np.log2(r.ranks + 2), 0)


def accuracy(r: Rankings, top_rating: int, ks: tuple[int, ...] = KS) -> dict[str, float]:
    """nDCG graduado sobre todas as interações; recall e MRR sobre as positivas."""
    liked = r.ranks[r.liked]
    metrics = {}
    for k in ks:
        metrics[f"ndcg@{k}"] = float(graded_dcg(r, top_rating, k).mean())
        metrics[f"recall@{k}"] = float((liked < k).mean())
    metrics["mrr"] = float((1 / (liked + 1)).mean())  # ranks are 0-based
    metrics["n_users"] = int(len(liked))              # positive (user, held-out item) rows
    metrics["n_ndcg"] = int(len(r.ranks))             # all rows used by the graded nDCG
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


def stratified(r: Rankings, groups: np.ndarray, top_rating: int, k: int = 10) -> dict[str, float]:
    """nDCG@k graduado por grupo de usuários (ou de alvos)."""
    gains = graded_dcg(r, top_rating, k)
    return {f"ndcg@{k}[{g}]": float(gains[groups == g].mean()) for g in np.unique(groups)}


def full_report(r: Rankings, popularity: np.ndarray, n_users: int,
                activity: np.ndarray, top_rating: int) -> dict[str, float]:
    target_group = np.where(head_items(popularity)[r.targets], "alvo cabeça", "alvo cauda")
    return {**accuracy(r, top_rating), **beyond_accuracy(r.subset(r.liked), popularity, n_users),
            **stratified(r, activity, top_rating), **stratified(r, target_group, top_rating)}


def activity_groups(rows: pd.DataFrame, user_count: dict) -> np.ndarray:
    """Faixas de interações no treino, onde se testa a hipótese do gate."""
    n = rows["user"].map(user_count).fillna(0).to_numpy()
    return np.select([n <= 1, n <= 3], ["1 interação", "2-3 interações"], "4+ interações")


# ---------- Protocolo ----------

class Protocol:
    """Tudo o que a avaliação precisa, calculado só com o treino (e treino + validação no teste).

    train / val / test: DataFrames com colunas user, item, rating (nota original).
    """

    def __init__(self, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame):
        train, val, test = (df[["user", "item", "rating"]] for df in (train, val, test))
        self.top_rating = rating_levels(train)
        thresholds, fallback = user_thresholds(train)
        train, val, test = (mark_liked(df, thresholds, fallback) for df in (train, val, test))

        self.items = sorted(train["item"].unique().tolist())
        self.catalog = {item: j for j, item in enumerate(self.items)}
        counts = train["item"].value_counts()
        self.popularity = counts.reindex(self.items).to_numpy(dtype=float)
        self.n_users = train["user"].nunique()
        self.user_count = train["user"].value_counts().to_dict()

        seen_train = train.groupby("user")["item"].agg(set).to_dict()
        seen_trainval = pd.concat([train, val]).groupby("user")["item"].agg(set).to_dict()
        self.history = {"val": seen_train, "test": seen_trainval}
        self.rows = {"val": evaluation_rows(val, self.catalog, seen_train),
                     "test": evaluation_rows(test, self.catalog, seen_trainval)}
        self.sizes = {"val": len(val), "test": len(test)}

    def describe(self) -> str:
        return " | ".join(f"{s}: {len(r)} avaliáveis ({int(r['liked'].sum())} positivas) de {self.sizes[s]}"
                          for s, r in self.rows.items()) + f" | catálogo: {len(self.items)} itens"

    def rank(self, score_fn: Callable[[list], torch.Tensor], split: str, chunk: int) -> Rankings:
        return rank_all(score_fn, self.rows[split], self.catalog, self.history[split], chunk)

    def accuracy(self, score_fn: Callable[[list], torch.Tensor], split: str, chunk: int) -> dict[str, float]:
        """Só as métricas de acurácia (seleção da época)."""
        return accuracy(self.rank(score_fn, split, chunk), self.top_rating)

    def report(self, score_fn: Callable[[list], torch.Tensor], split: str, chunk: int
               ) -> tuple[dict[str, float], Rankings]:
        """Relatório completo (acurácia, além da acurácia e estratificado) e os rankings por linha."""
        rankings = self.rank(score_fn, split, chunk)
        activity = activity_groups(self.rows[split], self.user_count)
        return full_report(rankings, self.popularity, self.n_users, activity, self.top_rating), rankings
