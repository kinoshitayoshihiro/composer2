import torch
import pandas as pd
import numpy as np
import pretty_midi as pm
from ml_models.pedal_model import PedalModel

def load_model(path):
    ckpt = torch.load(path, map_location="cpu")
    model = PedalModel()
    if "state_dict" in ckpt:
        state_dict = {k.replace("model.", ""): v for k, v in ckpt["state_dict"].items()}
        model.load_state_dict(state_dict)
    else:
        model.load_state_dict(ckpt)
    return model.eval()

# 以下、元のコードと同じ
CSV = "data/pedal/val.csv"
CKPT = "checkpoints/pedal.ckpt"
FPS = 100.0
THRESH = 0.5

df = pd.read_csv(CSV)
feat_cols = [c for c in df.columns if c.startswith("chroma_")] + ["rel_release"]
X = torch.tensor(df[feat_cols].values, dtype=torch.float32)

# ... 残りの処理 ...
