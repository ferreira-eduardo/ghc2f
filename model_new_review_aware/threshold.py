import pandas as pd


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
