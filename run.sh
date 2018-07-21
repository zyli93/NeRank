#!/usr/bin/env bash

COMPUTER=$(whoami)


if [ "$COMPUTER" == "ww8" ] ; then
    PY=/home/ww8/anaconda3/bin/python
elif [ "$COMPUTER" == "root" ] ; then
    PY=python
else
    PY=/home/zeyu/anaconda3/bin/python
fi

CUDA_VISIBLE_DEVICES=$1 $PY src/main.py --dataset $2 \
        --window-size 4 --neg-ratio 3 --embedding-dim 128 \
        --lstm-layers 1 --epoch-number 100 --batch-size 200 \
        --learning-rate 0.001 --cnn-channel 64 --lambda 100000 \
        --length $5 --coverage $4 \
        --precision_at_K 5 --id $3 --test-ratio 0.4 \
        #--include-content
        #--gen-metapaths --length 15 --coverage 3 --alpha 0.0 --metapaths "AQRQA" 
        #--preprocess --test-threshold 3 --proportion-test 0.1 --test-size 20\


