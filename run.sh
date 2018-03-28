#!/usr/bin/env bash

COMPUTER=$(whoami)

if [ "$COMPUTER" == "ww8" ] ; then
    PY=/home/ww8/anaconda3/bin/python
else
    PY=python3
fi

$PY src/main.py --dataset 3dprinting \
                    --length 15 \
                    --size 2 \
                    --alpha 0.0 \
                    --meta_paths "AQRQA AQA" \
                    --preprocess \

