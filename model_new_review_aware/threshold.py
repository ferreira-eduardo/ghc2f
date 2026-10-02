import pandas as pd


def user_thresholds(train: pd.DataFrame, k: float = 5.0,
                    low: float = 3.0, high: float = 4.0) -> tuple[pd.Series, float]:
    """Limiar de positivo por usuário: média encolhida para a global, recortada em [low, high].

    Calculado só com o treino. Devolve os limiares por usuário e o limiar global,
    usado para usuários que não aparecem no treino.
    """
    global_mean = train["rating"].mean()
    stats = train.groupby("user")["rating"].agg(["sum", "count"])
    shrunk = (stats["sum"] + k * global_mean) / (stats["count"] + k)
    fallback = min(max(global_mean, low), high)
    return shrunk.clip(lower=low, upper=high), fallback


def mark_liked(df: pd.DataFrame, thresholds: pd.Series, fallback: float) -> pd.DataFrame:
    """Acrescenta a coluna 'liked'. Aplicar com os mesmos limiares em treino, validação e teste."""
    limit = df["user"].map(thresholds).fillna(fallback)
    return df.assign(liked=df["rating"] >= limit)
