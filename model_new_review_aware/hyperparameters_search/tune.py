"""Busca de hiperparâmetros com Optuna.

Exemplos:
  python tune.py --variant full --stage high --trials 30 --root data/imdb
  python tune.py --variant cf   --stage high --trials 30 --root data/imdb
  python tune.py --variant full --stage medium --trials 30 --root data/imdb --base-study imdb-full-high
  python tune.py --variant full --stage high --trials 0 --root data/imdb   # só exporta o melhor

O estudo se chama {dataset}-{variant}-{stage} e fica em --storage.
Ao final, os melhores hiperparâmetros (conjunto completo) e as métricas de validação são salvos em {root}/tuning/{estudo}_best.json.
"""
import argparse
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

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


def make_objective(base: HParams, space: dict, root: str, encoder: TextEncoder | None,
                   val_sample: int | None = None):
    def objective(trial: optuna.Trial) -> float:
        hp = suggest(trial, base, space)

        def report(epoch: int, metrics: dict) -> None:
            trial.report(metrics["ndcg@10"], epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        metrics = main(hp, root=root, output=None, encoder=encoder, on_epoch=report,
                       val_sample=val_sample)
        for name, value in metrics.items():
            trial.set_user_attr(name, value)
        return metrics["ndcg@10"]

    return objective


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--variant", choices=VARIANTS, default="full")
    parser.add_argument("--stage", choices=SPACES, default="high")
    parser.add_argument("--trials", type=int, default=30,
                        help="total de tentativas do estudo (inclui as já feitas); 0 só exporta")
    parser.add_argument("--base-study", default=None,
                        help="estudo anterior cujos melhores parâmetros ficam fixos")
    parser.add_argument("--storage", default="sqlite:///optuna.db")
    parser.add_argument("--root", default="data/all_beauty")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-sample", type=int, default=None,
                        help="avalia só N usuários fixos da validação (acelera datasets grandes)")
    parser.add_argument("--export", default=None,
                        help="arquivo JSON para a melhor configuração (padrão: <estudo>.json)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    base = HParams(variant=args.variant)
    if args.base_study:
        previous = optuna.load_study(study_name=args.base_study, storage=args.storage)
        # a base do estudo anterior já traz as etapas antes dele; best_params só a etapa dele
        previous_base = previous.user_attrs.get("base", {"variant": args.variant})
        # a variante é a pedida: um estudo de "cf" pode partir do melhor do "full"
        base = replace(HParams(**previous_base), **previous.best_params, variant=args.variant)

    study = optuna.create_study(
        study_name=f"{args.variant}-{args.stage}",
        storage=args.storage,
        load_if_exists=True,  # rodar de novo continua o estudo
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
    )
    study.set_user_attr("base", base.to_dict())

    # --trials é o total do estudo: rodar de novo após uma interrupção só completa o que falta
    done = sum(t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
               for t in study.trials)
    remaining = max(args.trials - done, 0)
    print(f"estudo {study.study_name}: {done} tentativas feitas, {remaining} a rodar")
    if remaining:
        encoder = warm_encoder(args.root) if base.uses_text else None
        objective = make_objective(base, SPACES[args.stage], args.root, encoder, args.val_sample)
        study.optimize(objective, n_trials=remaining)

    if not any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
        sys.exit(f"estudo {study.study_name}: nenhuma tentativa completa, nada a exportar")

    print(f"\nmelhor nDCG@10: {study.best_value:.4f}")
    print("melhores parâmetros:", study.best_params)

    best = replace(base, **study.best_params)  # configuração completa, não só a etapa atual
    path = args.export or f"{study.study_name}.json"
    best.save(path, study=study.study_name, trial=study.best_trial.number,
              stage=args.stage, base_study=args.base_study, n_trials=len(study.trials),
              val_metrics=study.best_trial.user_attrs, val_sample=args.val_sample,
              exported=datetime.now().isoformat())
    print(f"configuração salva em {path}")