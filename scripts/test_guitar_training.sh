#!/bin/bash
# Quick test: Train Guitar only with 3 epochs to verify fixes

set -e

echo "=== Guitar Base Model Training Test ==="
echo "Mode: REGRESSION ONLY"
echo "Epochs: 3 (quick test)"
echo "Batch: 64 x 4 grad_accum = effective 256"
echo "Workers: 0 (macOS optimized)"
echo ""

PYTHONPATH=. .venv311/bin/python scripts/train_phrase.py \
  data/phrase_csv/guitar_train_raw.csv \
  data/phrase_csv/guitar_val_raw.csv \
  --epochs 3 \
  --out checkpoints/guitar_duv_raw_test \
  --arch transformer \
  --d_model 512 \
  --nhead 8 \
  --layers 4 \
  --batch-size 64 \
  --grad-accum 4 \
  --num-workers 0 \
  --lr 1e-4 \
  --duv-mode reg \
  --w-vel-reg 1.0 \
  --w-dur-reg 1.0 \
  --w-vel-cls 0.0 \
  --w-dur-cls 0.0 \
  --device mps \
  --save-best \
  --progress

echo ""
echo "✓ Test complete!"
echo ""
echo "Check for these indicators of success:"
echo "  ✓ train_batches/epoch > 10,000 (not 1)"
echo "  ✓ vel_mae shows actual values (not 0.000)"
echo "  ✓ dur_mae shows actual values (not 0.000)"
echo "  ✓ val_loss changes from 0.693147"
echo "  ✓ epoch time > 10 seconds (not 0.1s)"
