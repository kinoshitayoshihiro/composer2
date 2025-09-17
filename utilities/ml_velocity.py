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
        self.input_dim = input_dim
        if torch is None:
            self._dummy = True
        else:
            super().__init__()
            self._dummy = False
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
            # Ensure that an untrained model predicts the default velocity
            # (64) by zeroing the output weights. This keeps the unit tests
            # deterministic and provides a sensible starting point for
            # fine-tuning.
            nn.init.zeros_(self.fc_out.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc_in(x)
        h = self.encoder(h)
        return self.fc_out(h).squeeze(-1)

    @staticmethod
    def load(path: str):
        """
        Robust loader:
        - .ts / .torchscript → torch.jit.load (TorchScript)
        - others (.ckpt / .pt / .pth など) → torch.load (PyTorch/Lightning ckpt)
        これで Lightning ckpt に対して jit.load を呼んで落ちる問題を防ぐ。
        """
        p = str(path)
        if p.endswith((".ts", ".torchscript")):
            mod = torch.jit.load(p, map_location="cpu")
            mod.eval()
            return mod
        else:
            # Lightning/PyTorch ckpt を想定
            obj = torch.load(p, map_location="cpu")
            return obj

    def predict(self, ctx, *, cache_key: str | None = None):
        if torch is None or getattr(self, "_dummy", False):
            import numpy as np
            return np.full((ctx.shape[0],), 64.0, dtype=np.float32)

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
