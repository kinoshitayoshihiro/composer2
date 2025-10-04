# DUV Base Model Training - Optimization Guide

## 🎯 Applied Optimizations for macOS/MPS

### Problem: Heavy Machine Load
- Large batch size (256) → High MPS memory pressure
- Multiple workers → CPU contention on macOS
- Excessive logging → Terminal slowdown

### Solutions Applied

#### 1. Batch Size + Gradient Accumulation
```bash
--batch-size 64        # Reduced from 256
--grad-accum 4         # Accumulate 4 steps
# Effective batch size = 64 × 4 = 256 (same as before)
```
**Benefits:**
- 75% less MPS memory per step
- Same effective batch size for convergence
- Smoother GPU utilization

#### 2. Worker Count Optimization
```bash
--num-workers 0        # Single-threaded data loading
```
**Benefits:**
- No CPU worker contention on macOS
- Reduced context switching overhead
- Better for MPS devices

#### 3. Log Management
```bash
# Redirect to file, monitor with tail
> logs/guitar_duv_raw.log 2>&1

# Monitor live:
tail -f logs/guitar_duv_raw.log
```
**Benefits:**
- No terminal output bottleneck
- Persistent logs for analysis
- Clean console output

#### 4. Removed Debug Noise
- Commented out `DEBUG: torch is None` prints
- Reduces log volume by ~50,000 lines per epoch

## 📊 Expected Performance

### Before Optimization
- Batch processing: Slow, frequent stalls
- Memory: High MPS pressure
- Logs: Terminal overwhelmed
- **Result:** Training barely progresses

### After Optimization  
- Batch processing: Smooth, consistent
- Memory: 75% lower peak usage
- Logs: Clean, monitorable
- **Result:** Full training runs complete

## 🚀 Usage

### Quick Test (3 epochs)
```bash
./scripts/test_guitar_training.sh
```

### Full Training (All 5 instruments)
```bash
# Run in background
nohup ./scripts/train_all_base_raw_fixed.sh &

# Monitor progress
./scripts/monitor_training.sh

# Watch specific instrument
tail -f logs/guitar_duv_raw.log
```

### Monitor Training
```bash
# Summary of all instruments
./scripts/monitor_training.sh

# Live follow (Ctrl+C to exit)
tail -f logs/guitar_duv_raw.log
```

## ✅ Success Indicators

Check logs for:
- ✓ `train_batches/epoch=50016` (not 1!)
- ✓ `vel_mae` shows real values (not 0.000)
- ✓ `dur_mae` shows real values (not 0.000)  
- ✓ `val_loss` changes from 0.693147
- ✓ Epoch time: 30-120 seconds (not 0.1s)

## 📁 File Structure

```
scripts/
├── train_all_base_raw_fixed.sh   # Main: Train all 5 instruments
├── test_guitar_training.sh       # Quick test: Guitar only, 3 epochs
└── monitor_training.sh            # Progress summary

logs/
├── guitar_duv_raw.log
├── bass_duv_raw.log
├── piano_duv_raw.log
├── strings_duv_raw.log
└── drums_duv_raw.log

checkpoints/
├── guitar_duv_raw.best.ckpt
├── bass_duv_raw.best.ckpt
├── piano_duv_raw.best.ckpt
├── strings_duv_raw.best.ckpt
└── drums_duv_raw.best.ckpt
```

## 🔧 Technical Details

### Regression-Only Mode
```bash
--duv-mode reg         # Regression only (no classification)
--w-vel-reg 1.0        # Velocity regression weight
--w-dur-reg 1.0        # Duration regression weight
--w-vel-cls 0.0        # Classification OFF
--w-dur-cls 0.0        # Classification OFF
```

**Why:** Classification was broken (all labels = 0), regression works perfectly.

### File-Based Grouping Fix
- **Problem:** All `bar=0` → entire dataset = 1 group → 1 batch per epoch
- **Solution:** Group by file changes in PhraseDataset
- **Result:** Proper batching with 50,016 batches/epoch

### Dataset Stats
| Instrument | Train Phrases | Val Phrases | Unique Vels | Batches/Epoch |
|------------|---------------|-------------|-------------|---------------|
| Guitar     | 3,205,872     | 356,209     | 127         | 50,016        |
| Bass       | 3,123,319     | 347,036     | 127         | 48,802        |
| Piano      | 1,462,646     | 162,517     | 127         | 22,854        |
| Strings    | 1,949,881     | 216,654     | 127         | 30,467        |
| Drums      | 1,095,715     | 121,747     | 124         | 17,121        |

## 🎼 Data Sources
- **Guitar/Bass/Strings:** Los-Angeles-MIDI dataset (LAMD)
- **Piano:** POP909 dataset  
- **Drums:** loops dataset (75K+ loops)

All with **full 127-level velocity resolution** (vs previous 8-level quantized data).
