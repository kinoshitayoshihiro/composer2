# ML Velocity Model (Phase ML-10)

This document describes the Transformer based velocity prediction model used in Phase ML-10.

## Training

Use `scripts/train_velocity.py` with configuration `configs/velocity_model.yaml`.
The dataset is a CSV file where each row contains note context features followed by the target velocity.
Training uses PyTorch Lightning 2.3 and Hydra for configuration management. Early stopping
is triggered when validation MSE drops below `0.015` or when no improvement is seen for 10 epochs.
WandB logging can be enabled via the config file.

Example:
```bash
python scripts/train_velocity.py data=path/to/train.csv
```
A checkpoint `checkpoints/last.ckpt` will be written on completion.

## Re-training Procedure
1. Prepare a CSV dataset with columns representing note context and the final `velocity` column.
2. Edit `configs/velocity_model.yaml` if needed.
3. Run the training script as above.
4. Use the resulting checkpoint path as `ml_velocity_model_path` for generators.

## Preset Comparison

The repository includes a tiny script below to visualise the difference
between the learned curve and the legacy preset.

```python
import matplotlib.pyplot as plt
import pandas as pd
from utilities.ml_velocity import MLVelocityModel

model = MLVelocityModel.load("checkpoints/last.ckpt")
ctx = pd.read_csv("tests/data/velocity_mini.csv").iloc[:, :-1].values
pred = model.predict(ctx)
plt.plot(pred, label="ML")
plt.plot([64]*len(pred), label="Preset")
plt.legend()
plt.savefig("velocity_compare.png")
```

The figure will show how the Transformer model adapts velocities compared to the static preset curve.
