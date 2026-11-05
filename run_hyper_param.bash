#!/bin/bash
# Full hyperparameter pipeline (Stage 1 -> Stage 2 -> Stage 3) on the BERTopic
# topic distributions in results/topics/{dataset}/fold{k}/, one run per fold.
#
# Usage:  bash run_hyper_param.bash [dataset ...]      (default: amazon imdb)
#         MODELS="ghc2f" FOLDS="0 1" bash run_hyper_param.bash rotten_tomatoes
set -euo pipefail

cd "$(dirname "$0")"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

DATASETS="${*:-amazon imdb}"            # one or more dataset names
MODELS="${MODELS:-gated_hybrid ghc2f}"   # cf_autoencoder ignores text, so topics don't change it
FOLDS="0"

STORAGE="sqlite:///optuna.db"

STAGE1_TRIALS=60
STAGE2_TRIALS=50
EPOCHS=30          # per Stage 1/2 trial
STAGE3_EPOCHS=40
BATCH_SIZE=256

# Check every input before starting, so a missing fold doesn't fail hours in.
for DATASET in $DATASETS; do
  for fold in $FOLDS; do
    for f in "results/topics/${DATASET}/fold${fold}/review_topic_dist.npy" "splits/${DATASET}_fold${fold}.csv"; do
      [[ -f "$f" ]] || { echo "Missing $f" >&2; exit 1; }
    done
  done
done

for DATASET in $DATASETS; do
  TOPICS_DIR="results/topics/${DATASET}"
  OUT_ROOT="grid_search/best_hyperparameters/topics/${DATASET}"
  LOG_DIR="logs/topics/${DATASET}"
  mkdir -p "$LOG_DIR"

  for fold in $FOLDS; do
    for model in $MODELS; do
      out_dir="${OUT_ROOT}/fold${fold}"
      log="${LOG_DIR}/${model}_fold${fold}.log"
      mkdir -p "$out_dir"

      echo "=== ${DATASET} | ${model} | fold ${fold} | topics -> ${out_dir} ==="
      # "_topics" in the study name keeps these studies apart from the
      # review-embedding studies already in optuna.db.
      uv run python grid_search/optimizer_hyper.py "$DATASET" \
        --model "$model" \
        --fold "$fold" \
        --splits_dir splits \
        --embeddings_path "${TOPICS_DIR}/fold${fold}/review_topic_dist.npy" \
        --stage1_trials "$STAGE1_TRIALS" \
        --stage2_trials "$STAGE2_TRIALS" \
        --epochs "$EPOCHS" \
        --stage3_epochs "$STAGE3_EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --storage "$STORAGE" \
        --study_prefix "${DATASET}_${model}_topics_fold${fold}" \
        --out_dir "$out_dir" \
        2>&1 | tee "$log"
    done
  done
done

echo "Done. Best hyperparameters per fold: grid_search/best_hyperparameters/topics/{dataset}/fold*/{dataset}_{model}.csv"
