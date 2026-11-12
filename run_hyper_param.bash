#!/bin/bash
# Full hyperparameter pipeline (Stage 1 -> Stage 2 -> Stage 3) on the BERTopic
# topic distributions in results/topics/{dataset}/fold{k}/, one run per fold.
#
# Usage:  bash run_hyper_param.bash [dataset ...]      (default: amazon imdb)
#         MODELS="ghc2f" FOLDS="0 1" bash run_hyper_param.bash rotten_tomatoes
set -euo pipefail

cd "$(dirname "$0")"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

DATASETS="${*:-amazon}"            # one or more dataset names
MODELS="${MODELS:-gated_hybrid ghc2f}"   # cf_autoencoder ignores text, so topics don't change it
FOLDS="0"

STORAGE="sqlite:///optuna.db"

# Flat folder with {dataset}_{model}.csv, the layout run_experiment.py --hparams_dir reads.
# Separate from the old HR@10 results in grid_search/old_best_hyperparameters/topics/{dataset}/fold0/.
HPARAMS_DIR="grid_search/best_hyperparameters"

# One parameter file per dataset/model, shared by all 5 final folds -> tune on a single fold.
if [[ $(wc -w <<< "$FOLDS") -ne 1 ]]; then
  echo "FOLDS must be a single fold (got: $FOLDS); a second fold would overwrite ${HPARAMS_DIR}" >&2
  exit 1
fi

STAGE1_TRIALS=40
STAGE2_TRIALS=40
EPOCHS=50
STAGE3_EPOCHS=40
BATCH_SIZE=256
PATIENCE=5

# Delete and re-run existing studies (needed after changing epochs/fractions, which
# otherwise raise a mismatch error). Set RESTART=0 to resume an interrupted run instead.
RESTART="${RESTART:-1}"
RESTART_FLAG=""
[[ "$RESTART" == "1" ]] && RESTART_FLAG="--restart"

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
  LOG_DIR="logs/topics/${DATASET}"

  # Per-dataset settings. Rotten Tomatoes has ~1.5k evaluated users and val nDCG@10
  # around 0.005, so sampling users makes trial-to-trial differences mostly noise:
  # use every user in Stages 1/2, more patience, and more updates per epoch.
  STAGE1_FRAC=0.5
  STAGE2_FRAC=""            # empty -> optimizer default (0.6; 0.7 for rotten_tomatoes)
  DS_PATIENCE=$PATIENCE
  DS_BATCH_SIZE=$BATCH_SIZE
  if [[ "$DATASET" == "rotten_tomatoes" ]]; then
    STAGE1_FRAC=1.0
    STAGE2_FRAC=1.0
    DS_PATIENCE=10
    DS_BATCH_SIZE=64
  fi
  mkdir -p "$LOG_DIR"

  for fold in $FOLDS; do
    for model in $MODELS; do
      out_dir="${HPARAMS_DIR}"
      log="${LOG_DIR}/${model}_fold${fold}.log"
      mkdir -p "$out_dir"

      echo "=== ${DATASET} | ${model} | fold ${fold} | topics -> ${out_dir} ==="
      # "_topics" in the study name keeps these studies apart from the
      # review-embedding studies already in old_optuna.db.
      uv run python grid_search/optimizer_hyper.py "$DATASET" \
        --model "$model" \
        --fold "$fold" \
        --splits_dir splits \
        --embeddings_path "${TOPICS_DIR}/fold${fold}/review_topic_dist.npy" \
        --stage1_trials "$STAGE1_TRIALS" \
        --stage2_trials "$STAGE2_TRIALS" \
        --epochs "$EPOCHS" \
        --stage3_epochs "$STAGE3_EPOCHS" \
        --batch_size "$DS_BATCH_SIZE" \
        --patience "$DS_PATIENCE" \
        --stage1_frac "$STAGE1_FRAC" \
        ${STAGE2_FRAC:+--stage2_frac "$STAGE2_FRAC"} \
        $RESTART_FLAG \
        --storage "$STORAGE" \
        --study_prefix "${DATASET}_${model}_topics_fold${fold}" \
        --out_dir "$out_dir" \
        2>&1 | tee "$log"
    done
  done
done

echo "Done. Best hyperparameters: ${HPARAMS_DIR}/{dataset}_{model}.csv"
echo "Final runs:"
echo "  uv run python run_experiment.py --datasets ${DATASETS} --hparams_dir ${HPARAMS_DIR} \\"
echo "    --embeddings_path 'results/topics/{dataset}/fold{fold}/review_topic_dist.npy'"
echo "  (rotten_tomatoes was tuned with --batch_size 64 --patience 10; pass the same to run_experiment.py)"
