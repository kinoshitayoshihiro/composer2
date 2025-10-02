#!/bin/bash
# Train Base DUV Model v3 - Guitar
# Using existing phrase CSV data (33,228 train + 1,760 valid phrases)

set -e

echo "🎸 Training Guitar Base DUV Model v3..."
echo "Data: data/phrase_csv/guitar_train.csv (33,228 phrases)"
echo "Valid: data/phrase_csv/guitar_valid.csv (1,760 phrases)"
echo "Output: checkpoints/guitar_duv_v3.best.ckpt"
echo ""

cd "$(dirname "$0")/.."

PYTHONPATH=. .venv311/bin/python scripts/train_phrase.py \
  data/phrase_csv/guitar_train.csv \
  data/phrase_csv/guitar_valid.csv \
  --out checkpoints/guitar_duv_v3.best.ckpt \
  --logdir logs/guitar_duv_v3 \
  --arch transformer \
  --d_model 512 \
  --nhead 8 \
  --layers 4 \
  --max-len 128 \
  --dropout 0.1 \
  --epochs 20 \
  --batch-size 16 \
  --lr 0.0003 \
  --weight-decay 0.01 \
  --grad-clip 1.0 \
  --scheduler cosine \
  --warmup-steps 500 \
  --duv-mode both \
  --vel-bins 16 \
  --dur-bins 16 \
  --use-bar-beat \
  --w-boundary 1.0 \
  --w-vel-reg 1.0 \
  --w-dur-reg 1.0 \
  --w-vel-cls 0.5 \
  --w-dur-cls 0.5 \
  --early-stopping 5 \
  --save-every 1 \
  --device auto \
  --num-workers 0 \
  --progress

echo ""
echo "✅ Guitar base model training complete!"
echo "Checkpoint saved: checkpoints/guitar_duv_v3.best.ckpt"
