#!/bin/bash

BOTTLENECK=512
LAYERS=4096
LR=0.0001
EPOCHS=50
BATCH_SIZE=512
DIM=64
DROPOUT=0.2


# Setup Environment
export PYTHONPATH=$PYTHONPATH:$(pwd)
mkdir -p outputs/

# Run Training
echo "Starting GHC2F..."

python3 run_ae_bpr.py --bottleneck $BOTTLENECK --layers $LAYERS --lr $LR --dropout $DROPOUT --epochs $EPOCHS \
--batch_size=64 --embedding_dim $DIM --dataset=amazon

python3 run_ae_bpr.py --bottleneck $BOTTLENECK --layers $LAYERS --lr $LR --dropout $DROPOUT --epochs $EPOCHS \
--batch_size $BATCH_SIZE --embedding_dim $DIM --dataset=imdb

python3 run_ae_bpr.py --bottleneck $BOTTLENECK --layers $LAYERS --lr $LR --dropout $DROPOUT --epochs $EPOCHS \
--batch_size $BATCH_SIZE --embedding_dim $DIM --dataset=rotten_tomatoes



