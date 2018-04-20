#!/usr/bin/env bash

COMPUTER=$(whoami)


if [ "$COMPUTER" == "ww8" ] ; then
    PY=/home/ww8/anaconda3/bin/python
elif [ "$COMPUTER" == "root" ] ; then
    PY=python
else
    PY=/home/zeyu/anaconda3/bin/python
fi

$PY src/main.py --dataset superuser \
        --window-size 3 --neg-ratio 3.0 --embedding-dim 300 \
        --lstm-layers 3 --epoch-number 1000 --batch-size 5 --learning-rate 0.01 --cnn-channel 32\
        --gen-metapaths --length 15 --coverage 3 --alpha 0.0 --metapaths "AQRQA" \
        --preprocess \
        --test-threshold 4 --proportion-test 0.1


