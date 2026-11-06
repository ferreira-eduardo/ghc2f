#!/bin/bash

# Run Training
echo "Starting training..."

SEEDS="23 43 2"


#python3 run_experiment.py --layers $LAYERS --lr $LR --dropout $DROPOUT --epochs $EPOCHS \
#--batch_size 1024 --embedding_dim $DIM --dataset=imdb
#
#python3 run_experiment.py --layers $LAYERS --lr $LR --dropout $DROPOUT --epochs $EPOCHS \
#--batch_size 512 --embedding_dim $DIM --dataset=rotten_tomatoes

python3 model_new_review_aware/train.py --hparams=data/all_beauty/tuning/all_beauty-full-high_best.json --seed 23 43 2



