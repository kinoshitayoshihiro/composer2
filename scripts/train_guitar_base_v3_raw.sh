#!/bin/bash
set -e

PYTHONPATH=. .venv311/bin/python scripts/train_phrase.py \
  data/phrase_csv/guitar_train_raw.csv \
  data/phrase_csv/guitar_val_raw.csv \
  --epochs 20 \
  --out checkpoints/guitar_duv_v3_raw \
  --arch transformer \
  --d_model 512 \
  --nhead 8 \
  --layers 4 \
  --batch-size 128 \
  --lr 1e-4 \
  --duv-mode both \
  --w-vel-reg 1.0 \
  --w-dur-reg 1.0 \
  --device mps \
  --save-best \
  --progress
