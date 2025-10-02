#!/bin/bash
# Train all base DUV models with raw data (127 velocity levels)

set -e

EPOCHS=15
D_MODEL=512
NHEAD=8
LAYERS=4
BATCH_SIZE=256
LR=1e-4
DEVICE=mps

echo "=== Training All Base DUV Models (Raw Data) ==="
echo "Epochs: $EPOCHS, Batch Size: $BATCH_SIZE"
echo "Device: $DEVICE"
echo ""

# Guitar
echo ">>> [1/5] Training Guitar Base Model..."
PYTHONPATH=. .venv311/bin/python scripts/train_phrase.py \
  data/phrase_csv/guitar_train_raw.csv \
  data/phrase_csv/guitar_val_raw.csv \
  --epochs $EPOCHS \
  --out checkpoints/guitar_duv_raw \
  --arch transformer \
  --d_model $D_MODEL \
  --nhead $NHEAD \
  --layers $LAYERS \
  --batch-size $BATCH_SIZE \
  --lr $LR \
  --duv-mode both \
  --vel-bins 16 \
  --dur-bins 16 \
  --w-vel-reg 1.0 \
  --w-dur-reg 1.0 \
  --device $DEVICE \
  --save-best

echo ""
echo ">>> [2/5] Training Bass Base Model..."
PYTHONPATH=. .venv311/bin/python scripts/train_phrase.py \
  data/phrase_csv/bass_train_raw.csv \
  data/phrase_csv/bass_val_raw.csv \
  --epochs $EPOCHS \
  --out checkpoints/bass_duv_raw \
  --arch transformer \
  --d_model $D_MODEL \
  --nhead $NHEAD \
  --layers $LAYERS \
  --batch-size $BATCH_SIZE \
  --lr $LR \
  --duv-mode both \
  --vel-bins 16 \
  --dur-bins 16 \
  --w-vel-reg 1.0 \
  --w-dur-reg 1.0 \
  --device $DEVICE \
  --save-best

echo ""
echo ">>> [3/5] Training Piano Base Model..."
PYTHONPATH=. .venv311/bin/python scripts/train_phrase.py \
  data/phrase_csv/piano_train_raw.csv \
  data/phrase_csv/piano_val_raw.csv \
  --epochs $EPOCHS \
  --out checkpoints/piano_duv_raw \
  --arch transformer \
  --d_model $D_MODEL \
  --nhead $NHEAD \
  --layers $LAYERS \
  --batch-size $BATCH_SIZE \
  --lr $LR \
  --duv-mode both \
  --vel-bins 16 \
  --dur-bins 16 \
  --w-vel-reg 1.0 \
  --w-dur-reg 1.0 \
  --device $DEVICE \
  --save-best

echo ""
echo ">>> [4/5] Training Strings Base Model..."
PYTHONPATH=. .venv311/bin/python scripts/train_phrase.py \
  data/phrase_csv/strings_train_raw.csv \
  data/phrase_csv/strings_val_raw.csv \
  --epochs $EPOCHS \
  --out checkpoints/strings_duv_raw \
  --arch transformer \
  --d_model $D_MODEL \
  --nhead $NHEAD \
  --layers $LAYERS \
  --batch-size $BATCH_SIZE \
  --lr $LR \
  --duv-mode both \
  --vel-bins 16 \
  --dur-bins 16 \
  --w-vel-reg 1.0 \
  --w-dur-reg 1.0 \
  --device $DEVICE \
  --save-best

echo ""
echo ">>> [5/5] Training Drums Base Model..."
PYTHONPATH=. .venv311/bin/python scripts/train_phrase.py \
  data/phrase_csv/drums_train_raw.csv \
  data/phrase_csv/drums_val_raw.csv \
  --epochs $EPOCHS \
  --out checkpoints/drums_duv_raw \
  --arch transformer \
  --d_model $D_MODEL \
  --nhead $NHEAD \
  --layers $LAYERS \
  --batch-size $BATCH_SIZE \
  --lr $LR \
  --duv-mode both \
  --vel-bins 16 \
  --dur-bins 16 \
  --w-vel-reg 1.0 \
  --w-dur-reg 1.0 \
  --device $DEVICE \
  --save-best

echo ""
echo "✓ All base models training complete!"
echo "Checkpoints saved to checkpoints/*_duv_raw.best.ckpt"
