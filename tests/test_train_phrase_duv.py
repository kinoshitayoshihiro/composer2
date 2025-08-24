import csv
import logging
import re
from pathlib import Path

import torch

from scripts.train_phrase import train_model


def _write_csv(path: Path) -> None:
    rows = [
        {
            "pitch": 60,
            "velocity": 80,
            "duration": 0.5,
            "pos": 0,
            "boundary": 1,
            "bar": 0,
            "instrument": "piano",
            "velocity_bucket": 1,
            "duration_bucket": 0,
        },
        {
            "pitch": 62,
            "velocity": 70,
            "duration": 0.25,
            "pos": 1,
            "boundary": 0,
            "bar": 0,
            "instrument": "piano",
            "velocity_bucket": 0,
            "duration_bucket": 1,
        },
        {
            "pitch": 64,
            "velocity": 90,
            "duration": 0.5,
            "pos": 0,
            "boundary": 1,
            "bar": 1,
            "instrument": "piano",
            "velocity_bucket": 1,
            "duration_bucket": 0,
        },
        {
            "pitch": 65,
            "velocity": 75,
            "duration": 0.25,
            "pos": 1,
            "boundary": 0,
            "bar": 1,
            "instrument": "piano",
            "velocity_bucket": 0,
            "duration_bucket": 1,
        },
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "pitch",
                "velocity",
                "duration",
                "pos",
                "boundary",
                "bar",
                "instrument",
                "velocity_bucket",
                "duration_bucket",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _losses_from_caplog(caplog):
    pattern = re.compile(r"train_loss ([0-9.]+)")
    losses = []
    for r in caplog.records:
        m = pattern.search(r.message)
        if m:
            losses.append(float(m.group(1)))
    return losses


def test_train_phrase_reg(tmp_path, caplog):
    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    _write_csv(train_csv)
    _write_csv(val_csv)
    out = tmp_path / "model_reg.ckpt"
    with caplog.at_level(logging.INFO):
        train_model(
            train_csv,
            val_csv,
            epochs=2,
            arch="lstm",
            out=out,
            batch_size=1,
            d_model=32,
            max_len=4,
            duv_mode="reg",
            w_vel_reg=0.1,
            w_dur_reg=0.1,
        )
    losses = _losses_from_caplog(caplog)
    assert len(losses) >= 2 and losses[1] <= losses[0]
    state = torch.load(out)
    sd = state["model"]
    assert "head_vel_reg.weight" in sd
    assert "head_dur_reg.weight" in sd
    assert "head_vel_cls.weight" not in sd


def test_train_phrase_cls(tmp_path, caplog):
    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    _write_csv(train_csv)
    _write_csv(val_csv)
    out = tmp_path / "model_cls.ckpt"
    with caplog.at_level(logging.INFO):
        train_model(
            train_csv,
            val_csv,
            epochs=2,
            arch="lstm",
            out=out,
            batch_size=1,
            d_model=32,
            max_len=4,
            duv_mode="cls",
            vel_bins=2,
            dur_bins=2,
            w_vel_cls=0.1,
            w_dur_cls=0.1,
        )
    losses = _losses_from_caplog(caplog)
    assert len(losses) >= 2 and losses[1] <= losses[0]
    state = torch.load(out)
    sd = state["model"]
    assert "head_vel_cls.weight" in sd
    assert "head_dur_cls.weight" in sd
    assert "head_vel_reg.weight" not in sd
