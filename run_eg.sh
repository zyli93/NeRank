#!/usr/bin/env bash

COMPUTER=$(whoami)


if [ "$COMPUTER" == "ww8" ] ; then
    PY=/home/ww8/anaconda3/bin/python
elif [ "$COMPUTER" == "root" ] ; then
    PY=python
else
    PY=/home/zeyu/anaconda3/bin/python
fi

CUDA_VISIBLE_DEVICES=$1 $PY src/main.py --dataset 3dprinting \
        --window-size 5 --neg-ratio 5 --embedding-dim 128 \
        --lstm-layers 3 --epoch-number 100 --batch-size 10 \
        --learning-rate 0.01 --cnn-channel 32 --lambda 1.5 \
        --length 15 --coverage 3 \
        #--gen-metapaths --length 15 --coverage 3 --alpha 0.0 --metapaths "AQRQA" \
        #--preprocess --test-threshold 3 --proportion-test 0.1 --test-size 20\
        --precision_at_K 4


