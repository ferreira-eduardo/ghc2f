#!/bin/bash

BOTTLENECK=512
LAYERS=4096
LR=0.0001
EPOCHS=1
BATCH_SIZE=1024
DIM=64
DROPOUT=0.5


# Setup Environment
export PYTHONPATH=$PYTHONPATH:$(pwd)
mkdir -p outputs/

# Run Training
echo "Starting GHC2F..."

#python3 run_ae.py --bottleneck $BOTTLENECK --layers $LAYERS --lr $LR --dropout $DROPOUT --epochs $EPOCHS \
#--batch_size 128 --embedding_dim $DIM --dataset=amazon
#
#python3 run_ae.py --bottleneck $BOTTLENECK --layers $LAYERS --lr $LR --dropout $DROPOUT --epochs $EPOCHS \
#--batch_size 1024 --embedding_dim $DIM --dataset=imdb

python3 run_ae.py --bottleneck $BOTTLENECK --layers $LAYERS --lr $LR --dropout $DROPOUT --epochs $EPOCHS \
--batch_size 512 --embedding_dim $DIM --dataset=rotten_tomatoes



