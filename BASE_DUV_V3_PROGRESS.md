# Base DUV Model v3 Rebuild - Progress Report

## 🎯 Goal
Rebuild base DUV models with current PhraseTransformer architecture to fix compatibility issues.

## 📊 Training Data

| Instrument | Train Phrases | Valid Phrases | Total |
|-----------|--------------|--------------|-------|
| **Guitar** | 33,228 | 1,760 | 34,988 |
| **Bass** | 23,612 | 1,175 | 24,787 |
| **Keys** | 35,068 | 2,087 | 37,155 |
| **Strings** | 5,569 | 379 | 5,948 |
| **TOTAL** | **97,477** | **5,401** | **102,878** |

## 🏗️ Model Architecture

```yaml
Architecture: Transformer
d_model: 512
nhead: 8
layers: 4
max_len: 128
dropout: 0.1

Training:
  epochs: 20
  batch_size: 16
  lr: 0.0003
  scheduler: cosine
  warmup_steps: 500

DUV Mode: both (velocity + duration)
  vel_bins: 16
  dur_bins: 16
  use_bar_beat: true
```

## ⚙️ Training Status

### ✅ Completed
- [x] Created training scripts (train_guitar_base_v3.sh, train_all_base_v3.sh)
- [x] Verified train_phrase.py compatibility with phrase CSV data
- [x] Test run successful (1 epoch, limited batches)

### 🔄 In Progress
- [ ] **Guitar Base v3** - Training started (background)
  - Command: `./scripts/train_guitar_base_v3.sh`
  - Log: `logs/guitar_duv_v3_training.log`
  - Output: `checkpoints/guitar_duv_v3.best.ckpt`
  - Monitor: `tail -f logs/guitar_duv_v3_training.log`

### ⏳ Pending
- [ ] Bass Base v3
- [ ] Keys Base v3
- [ ] Strings Base v3
- [ ] LoRA retraining (4 instruments)
- [ ] Pipeline integration test

## 📝 Next Steps

1. **Monitor Guitar training** (~2-4 hours estimated)
   ```bash
   tail -f logs/guitar_duv_v3_training.log
   ```

2. **After Guitar completes**: Run batch training
   ```bash
   ./scripts/train_all_base_v3.sh  # Trains all remaining instruments
   ```

3. **Update LoRA configs** to use v3 base checkpoints
   ```yaml
   base_checkpoint: "checkpoints/{instrument}_duv_v3.best.ckpt"
   ```

4. **Retrain LoRA adapters** using new base models

5. **Verify integration** with test_lora_pipeline.py

## 🐛 Issues Fixed

| Issue | Status |
|-------|--------|
| Old checkpoint (9/28) incompatible with current code | ✅ Rebuilding |
| Missing feature extraction layers (pitch_emb, etc.) | ✅ New architecture includes all |
| Velocity changes = 0 in humanization | 🔄 Will fix with v3 |
| LoRA trained on broken base model | 🔄 Retraining scheduled |

## 💾 Expected Outputs

```
checkpoints/
├── guitar_duv_v3.best.ckpt   (target: ~35MB)
├── bass_duv_v3.best.ckpt     (target: ~35MB)
├── keys_duv_v3.best.ckpt     (target: ~35MB)
└── strings_duv_v3.best.ckpt  (target: ~35MB)
```

## 📊 Training Time Estimates

- Guitar: 2-4 hours (33K phrases, 20 epochs)
- Bass: 2-3 hours (24K phrases)
- Keys: 3-4 hours (35K phrases)  
- Strings: 1-2 hours (5.6K phrases)

**Total estimated time: 8-13 hours**

---
*Generated: 2025-10-02*
*Status: Guitar training in progress*
