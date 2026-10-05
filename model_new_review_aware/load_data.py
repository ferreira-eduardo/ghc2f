from pathlib import Path

import pandas as pd

from model_new_review_aware.data import Profiles, load_profiles
from model_new_review_aware.review_aware import ITEM_FIELDS, USER_FIELDS

# Valores que o LLM usou para indicar campo vazio. Confira as variantes nos seus dados.
EMPTY_VALUES = {"", "none", "n/a", "na", "null", "nan", "not specified", "not mentioned"}


def clean_field(value) -> str:
    """Texto do campo, ou "" se o LLM indicou ausência."""
    text = "" if value is None else str(value).strip()
    return "" if text.lower().rstrip(".") in EMPTY_VALUES else text


def read_split(root: Path, name: str) -> pd.DataFrame:
    df = pd.read_parquet(root / "splits" / f"{name}.parquet")
    return df.rename(columns={"userId": "user", "itemId": "item"})[["user", "item", "rating", "timestamp"]]


def read_profiles(path: Path, id_col: str, new_id: str, fields: tuple[str, ...]) -> Profiles:
    df = pd.read_parquet(path)
    df = df[df["parse_ok"]].copy()                 # descarta falhas de parsing
    df[new_id] = df[id_col].astype(int)            # mesmo tipo dos splits
    for f in fields:
        df[f] = df[f].map(clean_field)
    return load_profiles(df, new_id, fields)


def load_all_beauty(root: str | Path = "data/all_beauty"):
    """Splits com colunas user, item, rating, timestamp e perfis (gerados só com o treino)."""
    root = Path(root)
    train, val, test = (read_split(root, name) for name in ("train", "val", "test"))
    profiles = root / "profiles" / "train"
    user_profiles = read_profiles(profiles / "user_profile.parquet", "userId", "user", USER_FIELDS)
    item_profiles = read_profiles(profiles / "item_profile.parquet", "itemId", "item", ITEM_FIELDS)
    return train, val, test, user_profiles, item_profiles



if __name__ == "__main__":
    raw = pd.read_parquet("data/all_beauty/profiles/train/user_profile.parquet")
    for f in USER_FIELDS:
        print(f, raw[f].str.strip().str.lower().value_counts().head(8).to_dict())

    train, val, test, users, items = load_all_beauty()

    print("usuários do treino com perfil:", train["user"].isin(users).mean())
    print("itens do treino com perfil:", train["item"].isin(items).mean())

    # 3. Taxa de preenchimento por campo, depois da limpeza
    for name, profiles, fields in (("user", users, USER_FIELDS), ("item", items, ITEM_FIELDS)):
        for f in fields:
            print(name, f, sum(bool(p[f]) for p in profiles.values()) / len(profiles))

    # 4. A fração da validação que é avaliável
    print("val com item conhecido:", val["item"].isin(train["item"]).mean())   # ~0,44
    print("usuários com 1 interação no treino:", (train["user"].value_counts() == 1).mean())
