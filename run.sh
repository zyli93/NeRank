#!/usr/bin/env bash

COMPUTER=$(whoami)


if [ "$COMPUTER" == "ww8" ] ; then
    PY=/home/ww8/anaconda3/bin/python
else
    PY=/home/zeyu/anaconda3/bin/python
fi

$PY src/main.py --dataset 3dprinting \
        --window-size 7 --neg-ratio 3.0 --embedding-dim 300 \
        --lstm-layers 3 --epoch-number 1000 --batch-size 5
        # --preprocess \
        # --gen-metapaths --length 128 --coverage 10 --alpha 0.0 --metapaths "AQRQA" \


