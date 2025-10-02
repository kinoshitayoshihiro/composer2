#!/bin/bash
# Training script optimized for Google Colab with GPU

set -e

EPOCHS=15
D_MODEL=512
NHEAD=8
LAYERS=4
BATCH_SIZE=256        # GPU can handle larger batches
GRAD_ACCUM=1          # No need for grad accum with GPU
NUM_WORKERS=2         # Colab can use 2 workers
LR=1e-4
DEVICE=cuda           # GPU

echo "=== Training All Base DUV Models (Google Colab) ==="
echo "Mode: REGRESSION ONLY"
echo "Epochs: $EPOCHS, Batch Size: $BATCH_SIZE"
echo "Workers: $NUM_WORKERS, Device: $DEVICE"
echo ""

# Check GPU
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
echo ""

# Guitar
echo ">>> [1/5] Training Guitar Base Model..."
PYTHONPATH=. python scripts/train_phrase.py \
  data/phrase_csv/guitar_train_raw.csv \
  data/phrase_csv/guitar_val_raw.csv \
  --epochs $EPOCHS \
  --out checkpoints/guitar_duv_raw \
  --arch transformer \
  --d_model $D_MODEL \
  --nhead $NHEAD \
  --layers $LAYERS \
  --batch-size $BATCH_SIZE \
  --grad-accum $GRAD_ACCUM \
  --num-workers $NUM_WORKERS \
  --lr $LR \
  --duv-mode reg \
  --w-vel-reg 1.0 \
  --w-dur-reg 1.0 \
  --w-vel-cls 0.0 \
  --w-dur-cls 0.0 \
  --device $DEVICE \
  --save-best \
  --progress

echo ""
echo ">>> [2/5] Training Bass Base Model..."
PYTHONPATH=. python scripts/train_phrase.py \
  data/phrase_csv/bass_train_raw.csv \
  data/phrase_csv/bass_val_raw.csv \
  --epochs $EPOCHS \
  --out checkpoints/bass_duv_raw \
  --arch transformer \
  --d_model $D_MODEL \
  --nhead $NHEAD \
  --layers $LAYERS \
  --batch-size $BATCH_SIZE \
  --grad-accum $GRAD_ACCUM \
  --num-workers $NUM_WORKERS \
  --lr $LR \
  --duv-mode reg \
  --w-vel-reg 1.0 \
  --w-dur-reg 1.0 \
  --w-vel-cls 0.0 \
  --w-dur-cls 0.0 \
  --device $DEVICE \
  --save-best \
  --progress

echo ""
echo ">>> [3/5] Training Piano Base Model..."
PYTHONPATH=. python scripts/train_phrase.py \
  data/phrase_csv/piano_train_raw.csv \
  data/phrase_csv/piano_val_raw.csv \
  --epochs $EPOCHS \
  --out checkpoints/piano_duv_raw \
  --arch transformer \
  --d_model $D_MODEL \
  --nhead $NHEAD \
  --layers $LAYERS \
  --batch-size $BATCH_SIZE \
  --grad-accum $GRAD_ACCUM \
  --num-workers $NUM_WORKERS \
  --lr $LR \
  --duv-mode reg \
  --w-vel-reg 1.0 \
  --w-dur-reg 1.0 \
  --w-vel-cls 0.0 \
  --w-dur-cls 0.0 \
  --device $DEVICE \
  --save-best \
  --progress

echo ""
echo ">>> [4/5] Training Strings Base Model..."
PYTHONPATH=. python scripts/train_phrase.py \
  data/phrase_csv/strings_train_raw.csv \
  data/phrase_csv/strings_val_raw.csv \
  --epochs $EPOCHS \
  --out checkpoints/strings_duv_raw \
  --arch transformer \
  --d_model $D_MODEL \
  --nhead $NHEAD \
  --layers $LAYERS \
  --batch-size $BATCH_SIZE \
  --grad-accum $GRAD_ACCUM \
  --num-workers $NUM_WORKERS \
  --lr $LR \
  --duv-mode reg \
  --w-vel-reg 1.0 \
  --w-dur-reg 1.0 \
  --w-vel-cls 0.0 \
  --w-dur-cls 0.0 \
  --device $DEVICE \
  --save-best \
  --progress

echo ""
echo ">>> [5/5] Training Drums Base Model..."
PYTHONPATH=. python scripts/train_phrase.py \
  data/phrase_csv/drums_train_raw.csv \
  data/phrase_csv/drums_val_raw.csv \
  --epochs $EPOCHS \
  --out checkpoints/drums_duv_raw \
  --arch transformer \
  --d_model $D_MODEL \
  --nhead $NHEAD \
  --layers $LAYERS \
  --batch-size $BATCH_SIZE \
  --grad-accum $GRAD_ACCUM \
  --num-workers $NUM_WORKERS \
  --lr $LR \
  --duv-mode reg \
  --w-vel-reg 1.0 \
  --w-dur-reg 1.0 \
  --w-vel-cls 0.0 \
  --w-dur-cls 0.0 \
  --device $DEVICE \
  --save-best \
  --progress

echo ""
echo "✓ All base models training complete!"
echo "Checkpoints saved to checkpoints/*_duv_raw.best.ckpt"
