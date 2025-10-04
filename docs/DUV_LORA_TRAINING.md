# DUV LoRA Training Guide

## Overview

`scripts/train_duv_lora.py` enables LoRA (Low-Rank Adaptation) fine-tuning of DUV (Duration + Velocity) humanization models using YAML-based configuration.

## Architecture

```
Base DUV Model (frozen)
    ↓
LoRA Adapters (trainable, r=8, α=16)
    ↓
Instrument-specific humanization (guitar/bass/piano/drums)
```

## Quick Start

### 1. Prepare Base Model

Train a universal DUV base model (or skip if you already have one):

```bash
# Train base model using original train_duv.py
python scripts/train_duv.py \
  --csv-train data/universal_train.csv \
  --csv-valid data/universal_valid.csv \
  --stats-json data/universal_stats.json \
  --out checkpoints/duv_universal.ckpt \
  --epochs 20
```

### 2. Configure Instrument-Specific LoRA

Edit `config/duv/{instrument}_Lora.yaml`:

```yaml
# Base model
base_checkpoint: checkpoints/duv_universal.ckpt
base_frozen: true

# LoRA settings
lora:
  enable: true
  r: 8                    # Rank (4-16)
  alpha: 16               # Scaling factor
  dropout: 0.05
  target_modules:
    - "attn.q_proj"       # Query projection
    - "attn.v_proj"       # Value projection
    - "attn.out_proj"     # Output projection
    - "ff.in"             # Feed-forward input
    - "ff.out"            # Feed-forward output

# Data
manifest: manifests/lamd_guitar_enriched.jsonl
train_split: 0.9

# Training
trainer:
  epochs: 6
  batch_size: 64
  lr: 2.0e-4
  precision: bf16       # or "fp16" / "32"
```

### 3. Train LoRA Adapter

```bash
python scripts/train_duv_lora.py \
  --config config/duv/guitar_Lora.yaml \
  --devices auto \
  --num-workers 4
```

**Output:**
- `checkpoints/duv_guitar_lora/duv_lora_best.ckpt` (best checkpoint)
- `checkpoints/duv_guitar_lora/duv_lora_final.ckpt` (final checkpoint)
- `checkpoints/scalers/guitar_duv.json` (normalization stats)

### 4. Apply in Generation

The trained LoRA adapter is automatically loaded by `utilities/duv_apply.py`:

```python
from utilities.duv_apply import apply_duv_to_pretty_midi

# Apply DUV humanization
humanized_pm = apply_duv_to_pretty_midi(
    pm=original_midi,
    model_path="checkpoints/duv_guitar_lora/duv_lora_best.ckpt",
    intensity=0.9,
    include_regex=r"(?i)guitar",
)
```

## Training for All Instruments

```bash
# Guitar
python scripts/train_duv_lora.py --config config/duv/guitar_Lora.yaml

# Bass
python scripts/train_duv_lora.py --config config/duv/bass_Lora.yaml

# Piano
python scripts/train_duv_lora.py --config config/duv/piano_Lora.yaml

# Drums
python scripts/train_duv_lora.py --config config/duv/drums_Lora.yaml
```

## Configuration Reference

### Required Fields

| Field | Description | Example |
|-------|-------------|---------|
| `base_checkpoint` | Path to base DUV model | `checkpoints/duv_universal.ckpt` |
| `manifest` | JSONL manifest with enriched features | `manifests/lamd_guitar_enriched.jsonl` |
| `output_dir` | Output directory for checkpoints | `checkpoints/duv_guitar_lora` |

### LoRA Settings

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `r` | LoRA rank (dimension) | 4-16 |
| `alpha` | Scaling factor | 16-32 |
| `dropout` | LoRA dropout rate | 0.0-0.1 |
| `target_modules` | Layers to adapt | `["attn.*", "ff.*"]` |

**Trainable Parameters:**
- Full model: ~5M parameters
- LoRA (r=8): ~100K parameters (2% of full model)

### Feature Configuration

```yaml
features:
  inputs:
    - beat_pos          # Position in beat
    - beat_frac         # Fractional beat position
    - pitch             # MIDI pitch
    - dur_beats         # Duration in beats
    - prev_ioi          # Previous inter-onset interval
    - next_ioi          # Next inter-onset interval
    - is_downbeat       # Downbeat indicator
    - vel_norm          # Normalized velocity
    # Instrument-specific hints:
    - strum_dir_hint    # Guitar strum direction
    - section_onehot    # Section type (verse/chorus)
    - swing_hint        # Swing timing hint
  targets:
    - velocity          # Target velocity (0-127)
    - dur_beats         # Target duration (beats)
```

### Training Hyperparameters

```yaml
trainer:
  epochs: 6                    # Training epochs
  batch_size: 64               # Batch size
  lr: 2.0e-4                   # Learning rate
  weight_decay: 0.01           # L2 regularization
  warmup_steps: 1000           # LR warmup steps
  early_stop_patience: 2       # Early stopping patience
  grad_clip_norm: 1.0          # Gradient clipping
  precision: bf16              # Mixed precision (bf16/fp16/32)
  seed: 42                     # Random seed
```

## Data Requirements

### JSONL Manifest Format

Each line in the manifest should contain:

```json
{
  "file": "track_001.mid",
  "beat_pos": [0.0, 0.5, 1.0, ...],
  "pitch": [60, 62, 64, ...],
  "velocity": [80, 75, 90, ...],
  "dur_beats": [0.5, 0.5, 1.0, ...],
  "prev_ioi": [0.0, 0.5, 0.5, ...],
  "next_ioi": [0.5, 0.5, 1.0, ...],
  "is_downbeat": [1, 0, 0, ...],
  "strum_dir_hint": [1, 0, 1, ...]
}
```

### Filters

```yaml
filters:
  min_notes: 64              # Minimum notes per track
  max_duration_sec: 600      # Maximum track duration (10 min)
  min_duration_sec: 3        # Minimum track duration
```

## Inference Integration

### In BasePartGenerator

The DUV pipeline is automatically integrated:

```python
# generator/base_part_generator.py (already modified)
def compose(self, section_data):
    # ... generate MIDI ...
    
    # Apply Controls (onset micro-timing)
    if section_data['controls']['enable']:
        apply_guitar_controls(inst_pm, cfg)
        apply_piano_controls(inst_pm, cfg)
        apply_bass_controls(inst_pm, cfg)
        apply_drum_controls(inst_pm, cfg)
    
    # Apply DUV (velocity/duration humanization)
    if section_data.get('duv'):
        inst_pm = apply_duv_to_pretty_midi(
            pm=inst_pm,
            model_path=section_data['duv']['model_path'],
            intensity=section_data['duv'].get('intensity', 0.9),
        )
    
    return inst_pm
```

### Manual Usage

```python
from utilities.duv_apply import apply_duv_to_pretty_midi
import pretty_midi

# Load MIDI
pm = pretty_midi.PrettyMIDI("input.mid")

# Apply DUV
humanized_pm = apply_duv_to_pretty_midi(
    pm=pm,
    model_path="checkpoints/duv_guitar_lora/duv_lora_best.ckpt",
    intensity=0.9,              # 0.0-1.0 (blend with original)
    include_regex=r"(?i)guitar", # Track name filter
    exclude_regex=r"(?i)drum",   # Track name exclusion
)

# Save
humanized_pm.write("output.mid")
```

## Troubleshooting

### Issue: "No data loaded from manifest"

**Cause:** Manifest file is empty or malformed.

**Solution:**
```bash
# Verify manifest
head -n 1 manifests/lamd_guitar_enriched.jsonl | jq .

# Re-enrich if needed
python scripts/enrich_manifest.py \
  --input manifests/lamd_guitar.jsonl \
  --output manifests/lamd_guitar_enriched.jsonl
```

### Issue: "bf16 not available"

**Cause:** GPU doesn't support bfloat16 precision.

**Solution:** Change `precision: bf16` to `precision: fp16` or `precision: "32"` in YAML config.

### Issue: "Base checkpoint not found"

**Cause:** `checkpoints/duv_universal.ckpt` doesn't exist.

**Solution:** Train base model first (see Step 1) or set `base_checkpoint: null` to train from scratch.

### Issue: Low validation performance

**Solutions:**
1. Increase LoRA rank: `r: 16` (more capacity)
2. More training epochs: `epochs: 10`
3. Adjust learning rate: `lr: 1.0e-4` (slower)
4. Check data quality: Verify enriched features are correct

## Performance Tips

### Training Speed

- **Use GPU:** `--devices cuda` (10x faster than CPU)
- **Increase batch size:** `batch_size: 128` (if memory allows)
- **Mixed precision:** `precision: bf16` (faster on modern GPUs)
- **More workers:** `--num-workers 8` (faster data loading)

### Memory Optimization

If out of memory:
1. Reduce batch size: `batch_size: 32`
2. Lower precision: `precision: fp16`
3. Smaller LoRA rank: `r: 4`
4. Reduce max sequence length: `max_len: 128`

### Quality vs Speed Trade-off

| LoRA Rank | Training Time | Quality | Use Case |
|-----------|---------------|---------|----------|
| `r: 4` | Fast | Good | Quick experiments |
| `r: 8` | Medium | Better | Production (default) |
| `r: 16` | Slow | Best | High-quality humanization |

## Monitoring Training

### TensorBoard (Optional)

```bash
# Install tensorboard
pip install tensorboard

# View logs
tensorboard --logdir checkpoints/duv_guitar_lora/logs
```

### CSV Logs

Training metrics are saved to:
```
checkpoints/duv_guitar_lora/logs/version_0/metrics.csv
```

Columns:
- `train_loss`, `val_loss`: Total loss
- `train_vel_loss`, `val_vel_loss`: Velocity loss (MAE)
- `train_dur_loss`, `val_dur_loss`: Duration loss (Huber)

## Next Steps

1. ✅ Train LoRA adapters for all instruments
2. ✅ Integrate with BasePartGenerator (already done)
3. 🔄 Test end-to-end pipeline
4. 🔄 Evaluate humanization quality
5. 🔄 Fine-tune hyperparameters based on results

## References

- LoRA paper: https://arxiv.org/abs/2106.09685
- PhraseTransformer architecture: `models/phrase_transformer.py`
- DUV inference: `utilities/duv_infer.py`
- Controls integration: `utilities/guitar_controls.py`, `utilities/controls_bundle.py`
