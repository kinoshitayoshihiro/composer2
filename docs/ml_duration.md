# Duration Transformer

This model predicts note duration adjustments from bar-level sequences.

## Features
- `duration`
- `velocity`
- `pitch_class`
- `position_in_bar`

Each sample corresponds to a single bar (up to `max_len` notes). Sequences are padded with zeroes and accompanied by a mask tensor.

## Hyperparameters
| name | default |
|------|---------|
| d_model | 64 |
| batch_size | 32 |
| epochs | 5 |
| max_len | 16 |

## Usage
Train:
```bash
python scripts/train_duration.py --data path/to/durations.csv \
    --out model.ckpt --batch-size 32 --max-len 16
```

Use for inference:
```python
from utilities.duration_datamodule import DurationDataModule
from utilities.ml_duration import DurationTransformer, predict

from types import SimpleNamespace
cfg = SimpleNamespace(data=SimpleNamespace(csv='durations.csv'), batch_size=4, max_len=16)
dm = DurationDataModule(cfg)
model = DurationTransformer(max_len=cfg.max_len)
feats, _, mask = next(iter(dm.val_dataloader()))
preds = predict(feats, mask, model)
```

When adjusting `max_len`, pass the same value to `DurationTransformer(max_len=N)`.
The internal `PositionalEncoding` is created with this `max_len`, so ensure it
matches the longest expected bar length.

### Padding & CLS token
The datamodule prepends a learnable CLS token to each sequence. The padding mask
marks valid tokens with `True`; it is inverted before passing to the transformer
(`src_key_padding_mask = ~pad_mask`).
