#!/bin/bash
# Train Guitar Base Model with Raw Data - FIXED VERSION
# Changes:
# 1. duv-mode: both → reg (regression only, no classification)
# 2. Remove vel-bins/dur-bins (not needed for reg mode)
# 3. Explicitly set w-vel-cls=0, w-dur-cls=0
# 4. Add logging for batch counts

set -e

PYTHONPATH=. .venv311/bin/python scripts/train_phrase.py \
  data/phrase_csv/guitar_train_raw.csv \
  data/phrase_csv/guitar_val_raw.csv \
  --epochs 15 \
  --out checkpoints/guitar_duv_raw_fixed \
  --arch transformer \
  --d_model 512 \
  --nhead 8 \
  --layers 4 \
  --batch-size 256 \
  --lr 1e-4 \
  --duv-mode reg \
  --w-vel-reg 1.0 \
  --w-dur-reg 1.0 \
  --w-vel-cls 0.0 \
  --w-dur-cls 0.0 \
  --device mps \
  --save-best \
  --progress

echo "✓ Training complete"
