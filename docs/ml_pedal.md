# ML Pedal Model

This model predicts sustain pedal usage from frame‑level chroma features.

## Training

Use `scripts/train_pedal.py` with configuration `configs/pedal_model.yaml`.
The training CSV should contain the columns

```
track_id,frame_id,chroma_0,...,chroma_11,rel_release,pedal_state
```

`pedal_state` is `1` when the pedal is pressed and `0` when released.
Class imbalance can be adjusted via `class_weight` in the config file.

## Inference

```python
from utilities.ml_pedal import MLPedalModel, predict
model = MLPedalModel.load("checkpoints/last.ckpt")
cc_events = predict(score, model)
```

The returned list contains `(time, cc, value)` tuples representing CC64 events.
