import sys
from pathlib import Path
from types import ModuleType

import pytest

from .torch_stub import _stub_torch


_stub_torch()

pt = ModuleType("models.phrase_transformer")


class PhraseTransformer:  # pragma: no cover - simple stub
    """Lightweight stand-in that mimics the real transformer output shape."""

    def __init__(self, *args, **kwargs):
        d_model = kwargs.get("d_model") if "d_model" in kwargs else (args[0] if args else 16)
        max_len = kwargs.get("max_len") if "max_len" in kwargs else (
            args[1] if len(args) > 1 else 128
        )
        self.d_model = int(d_model)
        self.max_len = int(max_len)
        size = [[0.0] * self.max_len for _ in range(self.max_len)]
        self.pointer = size  # type: ignore[attr-defined]
        self.pointer_table = size  # type: ignore[attr-defined]
        self.pointer_bias = size  # type: ignore[attr-defined]

    def forward(self, feats=None, mask=None):  # pragma: no cover - simple stub
        try:
            import torch
        except Exception:  # torch absent in some environments
            torch = None  # type: ignore

        def _shape(obj):
            if torch is not None and isinstance(obj, torch.Tensor):
                if obj.dim() >= 2:
                    return int(obj.shape[0]), int(obj.shape[1])
                if obj.dim() == 1:
                    return 1, int(obj.shape[0])
            try:
                b = len(obj)
                inner = obj[0] if b else []
                t = len(inner) if isinstance(inner, (list, tuple)) else 1
                return int(b or 1), int(t or 1)
            except Exception:
                return None

        shape = _shape(mask)
        if shape is None and isinstance(feats, dict):
            for key in ("position", "pitch_class", "velocity", "duration"):
                shape = _shape(feats.get(key))
                if shape:
                    break
            if shape is None:
                for value in feats.values():
                    shape = _shape(value)
                    if shape:
                        break
        if shape is None:
            shape = (1, 1)

        bsz, seqlen = shape
        if torch is None:
            return [[0.0 for _ in range(seqlen)] for _ in range(bsz)]

        return torch.zeros(bsz, seqlen, dtype=torch.float32)

    __call__ = forward


pt.PhraseTransformer = PhraseTransformer
sys.modules.setdefault("models", ModuleType("models"))
sys.modules["models.phrase_transformer"] = pt

from scripts.train_phrase import train_model


def _write_csv(path: Path, rows: list[str]) -> None:
    path.write_text("pitch,velocity,duration,pos,boundary,bar\n" + "\n".join(rows))


def test_empty_training_dataset(tmp_path: Path) -> None:
    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    _write_csv(train_csv, [])
    _write_csv(val_csv, [])
    with pytest.raises(ValueError, match="training CSV produced no usable rows"):
        train_model(train_csv, val_csv, epochs=1, arch="lstm", out=tmp_path / "out.ckpt")


def test_empty_validation_dataset(tmp_path: Path) -> None:
    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    _write_csv(train_csv, ["60,64,1,0,0,1"])
    _write_csv(val_csv, [])
    with pytest.raises(ValueError, match="validation CSV produced no usable rows"):
        train_model(train_csv, val_csv, epochs=1, arch="lstm", out=tmp_path / "out.ckpt")

