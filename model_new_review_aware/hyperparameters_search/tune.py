"""Busca de hiperparâmetros com Optuna.

Exemplos:
  python tune.py --variant full --stage high --trials 30
  python tune.py --variant cf   --stage high --trials 30
  python tune.py --variant full --stage medium --trials 30 --base-study full-high
"""
import argparse
import sys
from dataclasses import replace
from pathlib import Path

# permite rodar como script (python .../tune.py) além de python -m
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import optuna
import torch

from model_new_review_aware.text_encoder import TextEncoder
from model_new_review_aware.load_data import load_all_beauty
from model_new_review_aware.hyperparameters_search.hyper_params import SPACES, VARIANTS, HParams, suggest
from model_new_review_aware.hyperparameters_search.train import main


def warm_encoder(root: str) -> TextEncoder:
    """Aquece o cache uma vez; todas as tentativas reaproveitam o mesmo encoder."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, _, users, items = load_all_beauty(root)
    encoder = TextEncoder().to(device)
    encoder.warm_up(t for p in (*users.values(), *items.values()) for t in p.values() if t.strip())
    return encoder


def make_objective(base: HParams, space: dict, root: str, encoder: TextEncoder | None):
    def objective(trial: optuna.Trial) -> float:
        hp = suggest(trial, base, space)

        def report(epoch: int, metrics: dict) -> None:
            trial.report(metrics["ndcg@10"], epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        metrics = main(hp, root=root, output=None, encoder=encoder, on_epoch=report)
        for name, value in metrics.items():
            trial.set_user_attr(name, value)
        return metrics["ndcg@10"]

    return objective


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--variant", choices=VARIANTS, default="full")
    parser.add_argument("--stage", choices=SPACES, default="high")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--base-study", default=None,
                        help="estudo anterior cujos melhores parâmetros ficam fixos")
    parser.add_argument("--storage", default="sqlite:///optuna.db")
    parser.add_argument("--root", default="data/all_beauty")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    base = HParams(variant=args.variant)
    if args.base_study:
        previous = optuna.load_study(study_name=args.base_study, storage=args.storage)
        base = replace(base, **previous.best_params)

    study = optuna.create_study(
        study_name=f"{args.variant}-{args.stage}",
        storage=args.storage,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
    )
    encoder = warm_encoder(args.root) if base.uses_text else None
    study.optimize(make_objective(base, SPACES[args.stage], args.root, encoder), n_trials=args.trials)

    print(f"\nmelhor nDCG@10: {study.best_value:.4f}")
    print("melhores parâmetros:", study.best_params)
    print("demais métricas:", study.best_trial.user_attrs)