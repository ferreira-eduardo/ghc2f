from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from review_aware import ITEM_FIELDS, USER_FIELDS, Batch

Profiles = dict[object, dict[str, str]]  # ID bruto -> {campo: texto}


def index_ids(values: pd.Series) -> dict:
    """IDs brutos -> índices a partir de 1. O 0 fica reservado para IDs fora do treino."""
    return {v: k for k, v in enumerate(pd.unique(values), start=1)}


def load_profiles(df: pd.DataFrame, id_col: str, fields: tuple[str, ...]) -> Profiles:
    """DataFrame de perfis -> dicionário indexado pelo ID bruto. Ausentes viram ""."""
    return df.set_index(id_col)[list(fields)].fillna("").astype(str).to_dict(orient="index")


def coverage(profile: dict | None, fields: tuple[str, ...]) -> float:
    """Fração de campos não vazios no perfil."""
    if not profile:
        return 0.0
    return sum(bool(profile.get(f, "").strip()) for f in fields) / len(fields)


@dataclass
class TrainStats:
    """Tudo o que é derivado do treino e reaproveitado em validação e teste."""
    user_index: dict
    item_index: dict
    user_count: dict
    item_count: dict
    item_log_q: dict
    user_mean: np.ndarray  # padronização das features do gate
    user_std: np.ndarray
    item_mean: np.ndarray
    item_std: np.ndarray

    @classmethod
    def from_train(cls, train: pd.DataFrame, user_profiles: Profiles,
                   item_profiles: Profiles) -> "TrainStats":
        user_count = train["user"].value_counts().to_dict()
        item_count = train["item"].value_counts().to_dict()
        total = len(train)
        item_log_q = {i: np.log(c / total) for i, c in item_count.items()}

        # Estatísticas sobre as linhas do treino: é a distribuição que o gate vê nos batches.
        u = np.array([cls._raw_user(user_count, user_profiles, x) for x in train["user"]])
        i = np.array([cls._raw_item(item_count, item_profiles, x) for x in train["item"]])

        return cls(
            user_index=index_ids(train["user"]), item_index=index_ids(train["item"]),
            user_count=user_count, item_count=item_count, item_log_q=item_log_q,
            user_mean=u.mean(0), user_std=u.std(0) + 1e-8,
            item_mean=i.mean(0), item_std=i.std(0) + 1e-8,
        )

    @staticmethod
    def _raw_user(counts: dict, profiles: Profiles, user) -> list[float]:
        return [np.log1p(counts.get(user, 0)), coverage(profiles.get(user), USER_FIELDS)]

    @staticmethod
    def _raw_item(counts: dict, profiles: Profiles, item) -> list[float]:
        return [np.log1p(counts.get(item, 0)), coverage(profiles.get(item), ITEM_FIELDS)]

    def user_features(self, user, profiles: Profiles) -> np.ndarray:
        return (self._raw_user(self.user_count, profiles, user) - self.user_mean) / self.user_std

    def item_features(self, item, profiles: Profiles) -> np.ndarray:
        return (self._raw_item(self.item_count, profiles, item) - self.item_mean) / self.item_std


class BatchBuilder:
    """collate_fn: lista de interações (dicts) -> Batch.

    Os perfis são os do split em uso; as estatísticas são sempre as do treino.
    """

    def __init__(self, stats: TrainStats, user_profiles: Profiles, item_profiles: Profiles):
        self.stats = stats
        self.user_profiles = user_profiles
        self.item_profiles = item_profiles

    def __call__(self, rows: list[dict]) -> Batch:
        s = self.stats
        users = [r["user"] for r in rows]
        items = [r["item"] for r in rows]

        def texts(profiles: Profiles, ids: list, fields: tuple[str, ...]) -> dict[str, list[str]]:
            return {f: [profiles.get(x, {}).get(f, "") for x in ids] for f in fields}

        def floats(values) -> torch.Tensor:
            return torch.tensor(np.asarray(values), dtype=torch.float32)

        return Batch(
            users=torch.tensor([s.user_index.get(u, 0) for u in users]),
            items=torch.tensor([s.item_index.get(i, 0) for i in items]),
            ratings=floats([r["rating"] for r in rows]),
            liked=torch.tensor([bool(r["liked"]) for r in rows]),
            item_log_q=floats([s.item_log_q.get(i, 0.0) for i in items]),
            user_texts=texts(self.user_profiles, users, USER_FIELDS),
            item_texts=texts(self.item_profiles, items, ITEM_FIELDS),
            user_features=floats([s.user_features(u, self.user_profiles) for u in users]),
            item_features=floats([s.item_features(i, self.item_profiles) for i in items]),
        )

if "__main__" == __name__:
    from torch.utils.data import DataLoader
    from threshold import user_thresholds, mark_liked

    
    thresholds, fallback = user_thresholds(train)
    train = mark_liked(train, thresholds, fallback)
    val = mark_liked(val, thresholds, fallback)

    train_users = load_profiles(train_user_df, "user", USER_FIELDS)
    train_items = load_profiles(train_item_df, "item", ITEM_FIELDS)
    stats = TrainStats.from_train(train, train_users, train_items)

    loader = DataLoader(
        train[["user", "item", "rating", "liked"]].to_dict("records"),
        batch_size=512, shuffle=True,
        collate_fn=BatchBuilder(stats, train_users, train_items),
    )
    batch = next(iter(loader))
    print(batch.users.shape, batch.user_features.shape, len(batch.user_texts["likes"]))