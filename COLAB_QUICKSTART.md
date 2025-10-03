# 🎯 Google Colab Quick Start Notebook

## Step 1: Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

## Step 2: Navigate to Project
```python
import os
os.chdir('/content/drive/Othercomputers/マイ MacBook Air/composer2-3')
!pwd
```

## Step 3: Verify GPU
```python
import torch
print(f"✓ CUDA: {torch.cuda.is_available()}")
print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
print(f"✓ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

## Step 4: Install Dependencies (if needed)
```python
!pip install -q pretty_midi pandas scikit-learn tqdm
```

## Step 5: Verify Data Files
```python
!ls -lh data/phrase_csv/*_raw.csv
```

Expected output:
```
guitar_train_raw.csv   261M
bass_train_raw.csv     ~250M
piano_train_raw.csv    87M
strings_train_raw.csv  ~180M
drums_train_raw.csv    88M
```

## Step 6A: Quick Test (3 epochs, 24 minutes)
```python
!PYTHONPATH=. python scripts/train_phrase.py \
  data/phrase_csv/guitar_train_raw.csv \
  data/phrase_csv/guitar_val_raw.csv \
  --epochs 3 \
  --out checkpoints/guitar_duv_raw_test \
  --arch transformer \
  --d_model 512 \
  --nhead 8 \
  --layers 4 \
  --batch-size 128 \
  --num-workers 2 \
  --lr 1e-4 \
  --duv-mode reg \
  --w-vel-reg 1.0 \
  --w-dur-reg 1.0 \
  --w-vel-cls 0.0 \
  --w-dur-cls 0.0 \
  --device cuda \
  --save-best \
  --progress
```

✅ Check for success:
```
INFO:root:[trainer] train_batches/epoch=25008 val_batches/epoch=2779
```

## Step 6B: Full Training (All 5 instruments, 7-8 hours)
```python
!bash scripts/train_all_colab.sh
```

⚠️ This will run for several hours. Monitor with:
```python
# In a separate cell
!tail -20 logs/guitar_duv_raw.log
```

## Alternative: Train One-by-One

### Guitar (2 hours)
```python
!PYTHONPATH=. python scripts/train_phrase.py \
  data/phrase_csv/guitar_train_raw.csv \
  data/phrase_csv/guitar_val_raw.csv \
  --epochs 15 \
  --out checkpoints/guitar_duv_raw \
  --arch transformer \
  --d_model 512 \
  --nhead 8 \
  --layers 4 \
  --batch-size 128 \
  --num-workers 2 \
  --lr 1e-4 \
  --duv-mode reg \
  --w-vel-reg 1.0 \
  --w-dur-reg 1.0 \
  --w-vel-cls 0.0 \
  --w-dur-cls 0.0 \
  --device cuda \
  --save-best \
  --progress
```

### Bass (2 hours)
```python
!PYTHONPATH=. python scripts/train_phrase.py \
  data/phrase_csv/bass_train_raw.csv \
  data/phrase_csv/bass_val_raw.csv \
  --epochs 15 \
  --out checkpoints/bass_duv_raw \
  --arch transformer \
  --d_model 512 \
  --nhead 8 \
  --layers 4 \
  --batch-size 128 \
  --num-workers 2 \
  --lr 1e-4 \
  --duv-mode reg \
  --w-vel-reg 1.0 \
  --w-dur-reg 1.0 \
  --w-vel-cls 0.0 \
  --w-dur-cls 0.0 \
  --device cuda \
  --save-best \
  --progress
```

### Piano (1 hour)
```python
!PYTHONPATH=. python scripts/train_phrase.py \
  data/phrase_csv/piano_train_raw.csv \
  data/phrase_csv/piano_val_raw.csv \
  --epochs 15 \
  --out checkpoints/piano_duv_raw \
  --arch transformer \
  --d_model 512 \
  --nhead 8 \
  --layers 4 \
  --batch-size 128 \
  --num-workers 2 \
  --lr 1e-4 \
  --duv-mode reg \
  --w-vel-reg 1.0 \
  --w-dur-reg 1.0 \
  --w-vel-cls 0.0 \
  --w-dur-cls 0.0 \
  --device cuda \
  --save-best \
  --progress
```

### Strings (1.5 hours)
```python
!PYTHONPATH=. python scripts/train_phrase.py \
  data/phrase_csv/strings_train_raw.csv \
  data/phrase_csv/strings_val_raw.csv \
  --epochs 15 \
  --out checkpoints/strings_duv_raw \
  --arch transformer \
  --d_model 512 \
  --nhead 8 \
  --layers 4 \
  --batch-size 128 \
  --num-workers 2 \
  --lr 1e-4 \
  --duv-mode reg \
  --w-vel-reg 1.0 \
  --w-dur-reg 1.0 \
  --w-vel-cls 0.0 \
  --w-dur-cls 0.0 \
  --device cuda \
  --save-best \
  --progress
```

### Drums (45 min)
```python
!PYTHONPATH=. python scripts/train_phrase.py \
  data/phrase_csv/drums_train_raw.csv \
  data/phrase_csv/drums_val_raw.csv \
  --epochs 15 \
  --out checkpoints/drums_duv_raw \
  --arch transformer \
  --d_model 512 \
  --nhead 8 \
  --layers 4 \
  --batch-size 128 \
  --num-workers 2 \
  --lr 1e-4 \
  --duv-mode reg \
  --w-vel-reg 1.0 \
  --w-dur-reg 1.0 \
  --w-vel-cls 0.0 \
  --w-dur-cls 0.0 \
  --device cuda \
  --save-best \
  --progress
```

## Monitor Training

```python
# Check GPU usage
!nvidia-smi

# Check training log
!tail -50 logs/guitar_duv_raw.log

# Check checkpoints
!ls -lh checkpoints/*_raw*
```

## After Training

Checkpoints will be saved to:
```
checkpoints/guitar_duv_raw.best.ckpt   (99 MB)
checkpoints/bass_duv_raw.best.ckpt     (99 MB)
checkpoints/piano_duv_raw.best.ckpt    (99 MB)
checkpoints/strings_duv_raw.best.ckpt  (99 MB)
checkpoints/drums_duv_raw.best.ckpt    (99 MB)
```

These will automatically sync to your MacBook Air via Google Drive!

## 🎉 Success!

Once training completes:
1. Wait for Google Drive sync (1-2 minutes per 99MB file)
2. On MacBook: Update LoRA configs
3. Retrain LoRA adapters
4. Test integrated pipeline
