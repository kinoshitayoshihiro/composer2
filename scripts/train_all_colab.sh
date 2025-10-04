#!/bin/bash
# Google Colab Training Script - All Instruments Sequential
# Expected total time: 7-8 hours on Colab T4, 3 hours on A100

set -e

PROJECT_DIR="/content/drive/Othercomputers/マイ MacBook Air/composer2-3"
cd "$PROJECT_DIR"

# Colab-optimized settings
BATCH_SIZE=128
NUM_WORKERS=2
DEVICE=cuda
EPOCHS=15
LR=1e-4
DUV_MODE=reg

echo "================================================"
echo "Google Colab Training - All Base DUV Models"
echo "================================================"
echo "Device: $DEVICE"
echo "Batch Size: $BATCH_SIZE"
echo "Workers: $NUM_WORKERS"
echo "Epochs: $EPOCHS"
echo "================================================"
echo ""

# Check CUDA
python3 << 'PY'
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("ERROR: No GPU found!")
    exit(1)
PY

echo ""
echo "================================================"
echo "Starting Training..."
echo "================================================"

# Function to train one instrument
train_instrument() {
    local INSTRUMENT=$1
    local START_TIME=$(date +%s)
    
    echo ""
    echo "================================================"
    echo "Training: ${INSTRUMENT} Base DUV Model"
    echo "Started: $(date)"
    echo "================================================"
    
    PYTHONPATH=. python scripts/train_phrase.py \
        "data/phrase_csv/${INSTRUMENT}_train_raw.csv" \
        "data/phrase_csv/${INSTRUMENT}_val_raw.csv" \
        --epochs $EPOCHS \
        --out "checkpoints/${INSTRUMENT}_duv_raw" \
        --arch transformer \
        --d_model 512 \
        --nhead 8 \
        --layers 4 \
        --batch-size $BATCH_SIZE \
        --num-workers $NUM_WORKERS \
        --lr $LR \
        --duv-mode $DUV_MODE \
        --w-vel-reg 1.0 \
        --w-dur-reg 1.0 \
        --w-vel-cls 0.0 \
        --w-dur-cls 0.0 \
        --device $DEVICE \
        --save-best \
        --progress
    
    local END_TIME=$(date +%s)
    local DURATION=$((END_TIME - START_TIME))
    local MINUTES=$((DURATION / 60))
    
    echo ""
    echo "✓ ${INSTRUMENT} training completed!"
    echo "  Duration: ${MINUTES} minutes"
    echo "  Checkpoint: checkpoints/${INSTRUMENT}_duv_raw.best.ckpt"
    
    # Check checkpoint size
    if [ -f "checkpoints/${INSTRUMENT}_duv_raw.best.ckpt" ]; then
        local SIZE=$(du -h "checkpoints/${INSTRUMENT}_duv_raw.best.ckpt" | cut -f1)
        echo "  Size: ${SIZE}"
    fi
    
    echo ""
}

# Train all instruments sequentially
TOTAL_START=$(date +%s)

train_instrument "guitar"
train_instrument "bass"
train_instrument "piano"
train_instrument "strings"
train_instrument "drums"

TOTAL_END=$(date +%s)
TOTAL_DURATION=$((TOTAL_END - TOTAL_START))
TOTAL_HOURS=$((TOTAL_DURATION / 3600))
TOTAL_MINUTES=$(((TOTAL_DURATION % 3600) / 60))

echo ""
echo "================================================"
echo "✓✓✓ ALL TRAINING COMPLETED! ✓✓✓"
echo "================================================"
echo "Total time: ${TOTAL_HOURS}h ${TOTAL_MINUTES}m"
echo ""
echo "Trained models:"
ls -lh checkpoints/*_duv_raw.best.ckpt
echo ""
echo "Next steps:"
echo "1. Wait for Google Drive to sync checkpoints to MacBook"
echo "2. Update LoRA configs to use new base models"
echo "3. Retrain LoRA adapters"
echo "4. Test integrated pipeline"
echo "================================================"
