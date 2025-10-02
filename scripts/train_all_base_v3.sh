#!/bin/bash
# Train all Base DUV Models v3
# Using existing phrase CSV data

set -e

echo "🎵 Training all Base DUV Models v3..."
echo "=================================="
echo ""

# Guitar: 33,228 train + 1,760 valid
echo "1️⃣  Training Guitar..."
./scripts/train_guitar_base_v3.sh

# Bass: 23,612 train + 1,175 valid  
echo ""
echo "2️⃣  Training Bass..."
cd "$(dirname "$0")/.."
PYTHONPATH=. .venv311/bin/python scripts/train_phrase.py \
  data/phrase_csv/bass_train.csv \
  data/phrase_csv/bass_valid.csv \
  --out checkpoints/bass_duv_v3.best.ckpt \
  --logdir logs/bass_duv_v3 \
  --arch transformer --d_model 512 --nhead 8 --layers 4 --max-len 128 --dropout 0.1 \
  --epochs 20 --batch-size 16 --lr 0.0003 --weight-decay 0.01 --grad-clip 1.0 \
  --scheduler cosine --warmup-steps 500 \
  --duv-mode both --vel-bins 16 --dur-bins 16 --use-bar-beat \
  --w-boundary 1.0 --w-vel-reg 1.0 --w-dur-reg 1.0 --w-vel-cls 0.5 --w-dur-cls 0.5 \
  --early-stopping 5 --save-every 1 --device auto --num-workers 0 --progress

# Keys: 35,068 train + 2,087 valid
echo ""
echo "3️⃣  Training Keys..."
PYTHONPATH=. .venv311/bin/python scripts/train_phrase.py \
  data/phrase_csv/keys_train.csv \
  data/phrase_csv/keys_valid.csv \
  --out checkpoints/keys_duv_v3.best.ckpt \
  --logdir logs/keys_duv_v3 \
  --arch transformer --d_model 512 --nhead 8 --layers 4 --max-len 128 --dropout 0.1 \
  --epochs 20 --batch-size 16 --lr 0.0003 --weight-decay 0.01 --grad-clip 1.0 \
  --scheduler cosine --warmup-steps 500 \
  --duv-mode both --vel-bins 16 --dur-bins 16 --use-bar-beat \
  --w-boundary 1.0 --w-vel-reg 1.0 --w-dur-reg 1.0 --w-vel-cls 0.5 --w-dur-cls 0.5 \
  --early-stopping 5 --save-every 1 --device auto --num-workers 0 --progress

# Strings: 5,569 train + 379 valid
echo ""
echo "4️⃣  Training Strings..."
PYTHONPATH=. .venv311/bin/python scripts/train_phrase.py \
  data/phrase_csv/strings_train.csv \
  data/phrase_csv/strings_valid.csv \
  --out checkpoints/strings_duv_v3.best.ckpt \
  --logdir logs/strings_duv_v3 \
  --arch transformer --d_model 512 --nhead 8 --layers 4 --max-len 128 --dropout 0.1 \
  --epochs 20 --batch-size 16 --lr 0.0003 --weight-decay 0.01 --grad-clip 1.0 \
  --scheduler cosine --warmup-steps 500 \
  --duv-mode both --vel-bins 16 --dur-bins 16 --use-bar-beat \
  --w-boundary 1.0 --w-vel-reg 1.0 --w-dur-reg 1.0 --w-vel-cls 0.5 --w-dur-cls 0.5 \
  --early-stopping 5 --save-every 1 --device auto --num-workers 0 --progress

echo ""
echo "=================================="
echo "✅ All Base DUV Models trained!"
echo ""
echo "Checkpoints:"
echo "  - checkpoints/guitar_duv_v3.best.ckpt"
echo "  - checkpoints/bass_duv_v3.best.ckpt"
echo "  - checkpoints/keys_duv_v3.best.ckpt"
echo "  - checkpoints/strings_duv_v3.best.ckpt"
