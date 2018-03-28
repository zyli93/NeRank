#!/usr/bin/env bash

COMPUTER=$(whoami)


if [ "$COMPUTER" == "ww8" ] ; then
    PY=/home/ww8/anaconda3/bin/python
else
    PY=/home/zeyu/anaconda3/bin/python
fi

$PY src/main.py --dataset 3dprinting \
        --preprocess \
        --gen-metapaths --length 15 --size 2 --alpha 0.0 --metapaths "AQRQA AQA"

