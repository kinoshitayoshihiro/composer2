from pathlib import Path
import csv

import pytest

pytest.importorskip("torch")

from scripts import train_phrase as tp


def test_duv_embed_toggle(tmp_path: Path) -> None:
    rows = [
        {
            "pitch": 60,
            "velocity": 64,
            "duration": 1,
            "pos": 0,
            "boundary": 1,
            "bar": 0,
            "instrument": "",
            "section": "",
            "mood": "",
            "velocity_bucket": 1,
            "duration_bucket": 2,
        }
    ]
    train_csv = tmp_path / "train.csv"
    tp.write_csv(rows, train_csv)
    valid_csv = tmp_path / "valid.csv"
    tp.write_csv(rows, valid_csv)
    ckpt1 = tmp_path / "a.ckpt"
    tp.train_model(
        train_csv,
        valid_csv,
        epochs=1,
        arch="lstm",
        out=ckpt1,
        batch_size=1,
        d_model=32,
        max_len=32,
        use_duv_embed=True,
    )
    assert ckpt1.is_file()
    ckpt2 = tmp_path / "b.ckpt"
    tp.train_model(
        train_csv,
        valid_csv,
        epochs=1,
        arch="lstm",
        out=ckpt2,
        batch_size=1,
        d_model=32,
        max_len=32,
        use_duv_embed=False,
    )
    assert ckpt2.is_file()

    # legacy short-column names
    rows_legacy = [
        {
            "pitch": 60,
            "velocity": 64,
            "duration": 1,
            "pos": 0,
            "boundary": 1,
            "bar": 0,
            "instrument": "",
            "section": "",
            "mood": "",
            "vel_bucket": 1,
            "dur_bucket": 2,
        }
    ]
    fields = [
        "pitch",
        "velocity",
        "duration",
        "pos",
        "boundary",
        "bar",
        "instrument",
        "section",
        "mood",
        "vel_bucket",
        "dur_bucket",
    ]
    train_csv2 = tmp_path / "train_old.csv"
    with train_csv2.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows_legacy)
    valid_csv2 = tmp_path / "valid_old.csv"
    with valid_csv2.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows_legacy)
    ckpt3 = tmp_path / "c.ckpt"
    tp.train_model(
        train_csv2,
        valid_csv2,
        epochs=1,
        arch="lstm",
        out=ckpt3,
        batch_size=1,
        d_model=32,
        max_len=32,
        use_duv_embed=True,
    )
    assert ckpt3.is_file()
