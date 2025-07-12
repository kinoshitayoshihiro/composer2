from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except Exception:  # pragma: no cover - optional
    torch = None  # type: ignore
    nn = object  # type: ignore

_script_cache: dict[str, Any] = {}


def _lru_script(model: nn.Module, key: str) -> Any:
    if torch is None:
        raise RuntimeError("torch required")
    cached = _script_cache.get(key)
    if cached is None:
        cached = torch.jit.script(model.eval())
        _script_cache[key] = cached
        if len(_script_cache) > 16:
            _script_cache.pop(next(iter(_script_cache)))
    return cached


def quantile_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float) -> torch.Tensor:
    diff = target - pred
    return torch.mean(torch.maximum(alpha * diff, (alpha - 1) * diff))


def velocity_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    huber = F.smooth_l1_loss(pred, target)
    q = quantile_loss(pred, target, 0.1) + quantile_loss(pred, target, 0.9)
    return huber + q


class ModelV1_LSTM(nn.Module if torch is not None else object):
    def __init__(self, input_dim: int = 3, hidden: int = 64) -> None:
        if torch is None:
            raise RuntimeError("torch required")
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden, 1)
        nn.init.constant_(self.fc.bias, 64.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out).squeeze(-1)


class MLVelocityModel(nn.Module if torch is not None else object):
    def __init__(self, input_dim: int = 3) -> None:
        if torch is None:
            raise RuntimeError("torch required")
        super().__init__()
        self.input_dim = input_dim
        self.fc_in = nn.Linear(input_dim, 256)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=4,
            dim_feedforward=256,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=4)
        self.fc_out = nn.Linear(256, 1)
        nn.init.constant_(self.fc_out.bias, 64.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc_in(x)
        h = self.encoder(h)
        return self.fc_out(h).squeeze(-1)

    @staticmethod
    def load(path: str) -> "MLVelocityModel":
        if torch is None:
            raise RuntimeError("torch required")
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict) and "state_dict" in obj:
            state = {k.replace("model.", ""): v for k, v in obj["state_dict"].items()}
            model = MLVelocityModel()
            model.load_state_dict(state, strict=False)
            model.eval()
            return model
        if isinstance(obj, dict) and all(isinstance(v, torch.Tensor) for v in obj.values()):
            model = MLVelocityModel()
            model.load_state_dict(obj, strict=False)
            model.eval()
            return model
        # assume scripted
        return torch.jit.load(path)

    def predict(self, ctx, *, cache_key: str | None = None):
        if torch is None:
            raise RuntimeError("torch required")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        x = torch.tensor(ctx, dtype=torch.float32, device=device).unsqueeze(0)
        model: Any = self
        if cache_key is not None:
            model = _lru_script(self, cache_key)
            model = model.to(device)
        else:
            model = self.to(device).eval()
        with torch.no_grad():
            out = model(x).squeeze(0).cpu().clamp(0, 127).to(torch.float32)
        return out.numpy()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Velocity model inference")
    parser.add_argument("--model", required=True)
    parser.add_argument("--json", required=True)
    args = parser.parse_args()

    data = json.loads(Path(args.json).read_text())
    model = MLVelocityModel.load(args.model)
    result = model.predict(data, cache_key=args.model)
    print(json.dumps(result.tolist()))


if __name__ == "__main__":
    main()

__all__ = [
    "MLVelocityModel",
    "ModelV1_LSTM",
    "velocity_loss",
]
