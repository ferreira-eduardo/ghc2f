from collections.abc import Callable
from dataclasses import asdict, dataclass, replace

import optuna

from model_new_review_aware.losses import LossWeights

VARIANTS = ("full", "cf", "text", "no_gate")


@dataclass(frozen=True)
class HParams:
    # Variante do modelo (ablação, não é ajustada)
    variant: str = "full"   # full | cf | text | no_gate

    # Treino
    lr: float = 1e-3
    batch_size: int = 512
    epochs: int = 50
    patience: int = 5       # fixo: o nDCG da validação oscila dentro do ruído
    seed: int = 42

    # Colaborativo
    cf_dim: int = 64
    cf_l2: float = 1e-4

    # Texto
    txt_dim: int = 128
    rank: int = 8
    matcher_l1: float = 1e-4

    # Perda (μ, α, λ: ajuste e análise; η: variável de estudo)
    mu_cf: float = 0.5
    mu_txt: float = 0.5
    alpha: float = 0.1
    eta: float = 0.1
    decorr: float = 0.1

    # Dados
    shrink_k: float = 5.0   # encolhimento do limiar por usuário

    def __post_init__(self):
        if self.variant not in VARIANTS:
            raise ValueError(f"variant deve ser um de {VARIANTS}, não {self.variant!r}")

    @property
    def uses_cf(self) -> bool:
        return self.variant != "text"

    @property
    def uses_text(self) -> bool:
        return self.variant != "cf"

    @property
    def uses_gate(self) -> bool:
        return self.variant == "full"

    def loss_weights(self) -> LossWeights:
        """Pesos da perda; termos de um ramo ausente são desligados."""
        both = self.uses_cf and self.uses_text
        return LossWeights(
            pair=self.alpha,
            ord=self.eta,
            sm_cf=self.mu_cf if both else 0.0,
            sm_txt=self.mu_txt if both else 0.0,
            decorr=self.decorr if both else 0.0,
            cf_l2=self.cf_l2 if self.uses_cf else 0.0,
            matcher_l1=self.matcher_l1 if self.uses_text else 0.0,
        )

    def to_dict(self) -> dict:
        return asdict(self)


# ---------- Espaços de busca (Optuna) ----------

Suggestion = Callable[[optuna.Trial, str], object]


def log_float(low: float, high: float) -> Suggestion:
    return lambda trial, name: trial.suggest_float(name, low, high, log=True)


def linear_float(low: float, high: float) -> Suggestion:
    return lambda trial, name: trial.suggest_float(name, low, high)


def categorical(*values) -> Suggestion:
    return lambda trial, name: trial.suggest_categorical(name, list(values))


# Etapa 1: os parâmetros a que o resultado é mais sensível.
HIGH = {
    "lr": log_float(1e-4, 3e-3),
    "batch_size": categorical(256, 512, 1024, 2048),
    "cf_dim": categorical(16, 32, 64, 128),
    "cf_l2": log_float(1e-4, 1e-1),
}

# Etapa 2: com os melhores da etapa 1 fixados.
MEDIUM = {
    "txt_dim": categorical(32, 64, 128, 256),
    "mu_cf": linear_float(0.0, 1.0),
    "mu_txt": linear_float(0.0, 1.0),
    "alpha": linear_float(0.0, 0.3),
}

# Etapa 3: ajuste fino, se houver orçamento.
LOW = {
    "rank": categorical(4, 8, 16),
    "matcher_l1": log_float(1e-5, 1e-2),
    "decorr": linear_float(0.0, 1.0),
    "shrink_k": categorical(2.0, 5.0, 10.0),
}

SPACES = {"high": HIGH, "medium": MEDIUM, "low": LOW}

# Parâmetros que só fazem sentido quando o ramo correspondente existe.
CF_ONLY = {"cf_dim", "cf_l2"}
TEXT_ONLY = {"txt_dim", "rank", "matcher_l1"}
BOTH_ONLY = {"mu_cf", "mu_txt", "decorr"}


def relevant(name: str, hp: HParams) -> bool:
    if name in CF_ONLY:
        return hp.uses_cf
    if name in TEXT_ONLY:
        return hp.uses_text
    if name in BOTH_ONLY:
        return hp.uses_cf and hp.uses_text
    return True


def suggest(trial: optuna.Trial, base: HParams, space: dict[str, Suggestion]) -> HParams:
    """Pede ao Optuna os parâmetros do espaço relevantes para a variante; os demais vêm de base."""
    values = {name: make(trial, name) for name, make in space.items() if relevant(name, base)}
    return replace(base, **values)