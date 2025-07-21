import torch
import pandas as pd
from utilities.ml_duration import DurationTransformer

# 学習時と同じパラメータでモデルを作成
model = DurationTransformer(d_model=64, max_len=16)
ckpt = torch.load("checkpoints/duration_earlystop.ckpt", map_location="cpu")
model.load_state_dict(ckpt["state_dict"])
model.eval()

# 新しいデータの読み込み例
df = pd.read_csv("data/your_test_data.csv")  # 推論したいデータ
# 必要に応じてデータ前処理

# 例: 1バーパートだけ推論
sample = df[df["bar"] == 0].sort_values("position")
import numpy as np
import torch

duration_input = torch.tensor(
    sample["duration"].tolist() + [0.0] * (16 - len(sample)), dtype=torch.float32
).unsqueeze(0)
velocity_input = torch.tensor(
    sample["velocity"].tolist() + [0.0] * (16 - len(sample)), dtype=torch.float32
).unsqueeze(0)
pitch_class_input = torch.tensor(
    (sample["pitch"] % 12).tolist() + [0] * (16 - len(sample)), dtype=torch.long
).unsqueeze(0)
position_input = torch.tensor(
    sample["position"].tolist() + [0] * (16 - len(sample)), dtype=torch.long
).unsqueeze(0)
mask = torch.zeros(16, dtype=torch.bool)
mask[: len(sample)] = 1
mask = mask.unsqueeze(0)

# モデルによって入力の形が異なる場合は合わせてください
with torch.no_grad():
    pred = model(
        duration_input, velocity_input, pitch_class_input, position_input, mask
    )
print("予測duration:", pred)
