import sys
from pathlib import Path
from types import ModuleType

import pytest

from .torch_stub import _stub_torch


_stub_torch()

pt = ModuleType("models.phrase_transformer")


class PhraseTransformer:  # pragma: no cover - simple stub
    pass


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

