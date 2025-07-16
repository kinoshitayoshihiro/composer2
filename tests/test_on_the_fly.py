import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "train_velocity.py"
    spec = importlib.util.spec_from_file_location("train_velocity", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore
    return module


def test_on_the_fly_augmentation(tmp_path: Path, monkeypatch) -> None:
    module = load_module()
    csv = tmp_path / "v.csv"
    pd.DataFrame({
        "pitch": [60],
        "duration": [1.0],
        "prev_vel": [64],
        "velocity": [70],
    }).to_csv(csv, index=False)

    noise = np.array([0.1, 0.1, 0.1], dtype=np.float32)

    def fake_normal(*args, **kwargs):
        return noise

    monkeypatch.setattr(module.np.random, "normal", fake_normal)

    class DummyTorch:
        @staticmethod
        def tensor(x):
            return x

    module.torch = DummyTorch()

    ds = module.CsvDataset(csv, 3, transform=lambda x: x + module.np.random.normal(scale=0.01, size=x.shape))
    x, y = ds[0]
    assert np.allclose(x, module.np.array([60, 1.0, 64]) + noise)
    assert y == 70
