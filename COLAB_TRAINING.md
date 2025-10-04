# Google Colab Training Setup for DUV Base Models

## 🚀 Quick Start

### 1. Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### 2. Navigate to Project
```bash
%cd /content/drive/Othercomputers/マイ MacBook Air/composer2-3
```

### 3. Install Dependencies (if needed)
```bash
!pip install -q torch torchvision torchaudio
!pip install -q pretty_midi pandas scikit-learn tqdm
```

### 4. Verify GPU
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

## 📊 Training Commands

### Quick Test (Guitar, 3 epochs)
```bash
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

### Full Training - Guitar (15 epochs)
```bash
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

### Full Training - Bass (15 epochs)
```bash
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

### Full Training - Piano (15 epochs)
```bash
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

### Full Training - Strings (15 epochs)
```bash
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

### Full Training - Drums (15 epochs)
```bash
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

## 🔧 Colab-Optimized Settings

### GPU Settings (vs macOS MPS)
| Setting | macOS | Colab GPU |
|---------|-------|-----------|
| `--device` | `mps` | **`cuda`** |
| `--batch-size` | `64` | **`128`** (2x) |
| `--num-workers` | `0` | **`2`** |
| `--grad-accum` | `4` | Not needed |

### Why Different?
- **Larger batch**: Colab GPU has more VRAM than MPS
- **Workers**: Colab has better multi-core support
- **No grad accum**: Larger batch fits in memory

## 📊 Expected Performance

### Training Speed (per epoch)
| Instrument | Batches | Colab T4 | Colab A100 |
|------------|---------|----------|------------|
| Guitar | 25,008 | ~8 min | ~3 min |
| Bass | 24,401 | ~8 min | ~3 min |
| Piano | 11,427 | ~4 min | ~1.5 min |
| Strings | 15,233 | ~5 min | ~2 min |
| Drums | 8,560 | ~3 min | ~1 min |

### Total Training Time (15 epochs)
| Instrument | Colab T4 | Colab A100 |
|------------|----------|------------|
| Guitar | ~2 hours | ~45 min |
| Bass | ~2 hours | ~45 min |
| Piano | ~1 hour | ~22 min |
| Strings | ~1.5 hours | ~30 min |
| Drums | ~45 min | ~15 min |
| **TOTAL** | **~7-8 hours** | **~3 hours** |

## ✅ Success Indicators

Check logs for:
```
INFO:root:[trainer] train_batches/epoch=25008 val_batches/epoch=2779
INFO:root:epoch 1 train_loss 1.234 vel_mae 12.45 dur_mae 0.034 time 480.2s
```

- ✓ `train_batches/epoch > 10,000`
- ✓ `vel_mae` decreasing (not 0.000)
- ✓ `dur_mae` decreasing (not 0.000)
- ✓ `val_loss` changing from 0.693147
- ✓ Epoch time: 3-10 minutes

## 💾 Checkpoints

Trained models will be saved to:
```
/content/drive/Othercomputers/マイ MacBook Air/composer2-3/checkpoints/
├── guitar_duv_raw.best.ckpt   (99 MB)
├── bass_duv_raw.best.ckpt     (99 MB)
├── piano_duv_raw.best.ckpt    (99 MB)
├── strings_duv_raw.best.ckpt  (99 MB)
└── drums_duv_raw.best.ckpt    (99 MB)
```

These will automatically sync to your MacBook Air via Google Drive!

## 🎯 Recommended Workflow

### Option A: One-by-One (Safer)
1. Train Guitar first (2 hours)
2. Verify checkpoint synced to MacBook
3. Train Bass (2 hours)
4. Train Piano (1 hour)
5. Train Strings (1.5 hours)
6. Train Drums (45 min)

### Option B: Sequential Script
```bash
# Run all in sequence (7-8 hours total)
!bash scripts/train_all_colab.sh
```

### Option C: Manual Control
Run each command above one-by-one in separate cells.
This allows you to:
- Monitor each instrument's progress
- Pause between trainings
- Verify checkpoints before continuing

## 🔍 Monitoring

### Check GPU Usage
```python
!nvidia-smi
```

### Check Training Progress
```bash
!tail -50 logs/guitar_duv_raw.log
```

### Check Checkpoint Size
```bash
!ls -lh checkpoints/*_raw*
```

## ⚠️ Important Notes

1. **Google Drive Sync**: Large files (99MB checkpoints) may take 1-2 minutes to sync
2. **Colab Timeout**: Free tier disconnects after 12 hours. Use Colab Pro for longer sessions
3. **Session Management**: Save checkpoints frequently (handled automatically with `--save-best`)
4. **Runtime Type**: Use GPU runtime (Runtime → Change runtime type → GPU)

## 🚨 If Training Interrupted

### Resume from Checkpoint (if available)
```bash
!PYTHONPATH=. python scripts/train_phrase.py \
  data/phrase_csv/guitar_train_raw.csv \
  data/phrase_csv/guitar_val_raw.csv \
  --resume checkpoints/guitar_duv_raw.ckpt \
  --epochs 15 \
  ... (other args same as above)
```

### Restart from Scratch
Just re-run the training command. Old checkpoints will be overwritten.
