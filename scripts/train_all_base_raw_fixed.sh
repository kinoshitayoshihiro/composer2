#!/bin/bash
# Train All Base DUV Models with Raw Data - FIXED VERSION
# 
# Fixes applied:
# 1. duv-mode: both → reg (regression only, no broken classification)
# 2. Remove vel-bins/dur-bins (not needed for reg mode)
# 3. Explicitly set w-vel-cls=0, w-dur-cls=0 to disable classification loss
# 4. PhraseDataset now groups by file instead of bar=0
# 5. Added batch count logging

set -e

EPOCHS=15
D_MODEL=512
NHEAD=8
LAYERS=4
BATCH_SIZE=64          # Reduced from 256 to ease MPS load
GRAD_ACCUM=4           # Accumulate 4 batches = effective batch size 256
NUM_WORKERS=0          # macOS: 0 workers to avoid CPU contention
LR=1e-4
DEVICE=mps

echo "=== Training All Base DUV Models (Raw Data - FIXED) ==="
echo "Mode: REGRESSION ONLY (no classification)"
echo "Epochs: $EPOCHS, Batch Size: $BATCH_SIZE (x$GRAD_ACCUM grad accum = effective 256)"
echo "Workers: $NUM_WORKERS, Device: $DEVICE"
echo "Logs will be written to logs/*_duv_raw.log"
echo ""

# Guitar
echo ">>> [1/5] Training Guitar Base Model..."
echo "    Log: logs/guitar_duv_raw.log (use 'tail -f logs/guitar_duv_raw.log' to monitor)"
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
  --progress > logs/guitar_duv_raw.log 2>&1

echo "    ✓ Guitar training complete"

echo ""
echo ">>> [2/5] Training Bass Base Model..."
echo "    Log: logs/bass_duv_raw.log"
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
  --progress > logs/bass_duv_raw.log 2>&1

echo "    ✓ Bass training complete"

echo ""
echo ">>> [3/5] Training Piano Base Model..."
echo "    Log: logs/piano_duv_raw.log"
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
  --progress > logs/piano_duv_raw.log 2>&1

echo "    ✓ Piano training complete"

echo ""
echo ">>> [4/5] Training Strings Base Model..."
echo "    Log: logs/strings_duv_raw.log"
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
  --progress > logs/strings_duv_raw.log 2>&1

echo "    ✓ Strings training complete"

echo ""
echo ">>> [5/5] Training Drums Base Model..."
echo "    Log: logs/drums_duv_raw.log"
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
  --progress > logs/drums_duv_raw.log 2>&1

echo "    ✓ Drums training complete"

echo ""
echo "✓ All base models training complete!"
echo "Checkpoints saved to checkpoints/*_duv_raw.best.ckpt"
echo ""
echo "To monitor training progress:"
echo "  tail -f logs/guitar_duv_raw.log"
echo "  tail -f logs/bass_duv_raw.log"
echo "  tail -f logs/piano_duv_raw.log"
echo "  tail -f logs/strings_duv_raw.log"
echo "  tail -f logs/drums_duv_raw.log"
echo ""
echo "Expected log patterns:"
echo "  - train_batches/epoch: 10,000+"
echo "  - vel_mae / dur_mae: non-zero and decreasing"
echo "  - val_loss: moving away from 0.693147"
echo "  - epoch time: 30s - several minutes (not 0.1s)"
