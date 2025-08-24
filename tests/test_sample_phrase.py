import csv
from pathlib import Path

import pretty_midi

from scripts.train_phrase import train_model
from scripts.sample_phrase import main as sample_main


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


def test_sample_phrase(tmp_path):
    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    _write_csv(train_csv)
    _write_csv(val_csv)
    ckpt = tmp_path / "model.ckpt"
    train_model(
        train_csv,
        val_csv,
        epochs=1,
        arch="lstm",
        out=ckpt,
        batch_size=1,
        d_model=32,
        max_len=4,
        duv_mode="reg",
        w_vel_reg=0.1,
        w_dur_reg=0.1,
    )
    midi_out = tmp_path / "out.mid"
    sample_main(
        [
            "--ckpt",
            str(ckpt),
            "--in",
            str(val_csv),
            "--arch",
            "lstm",
            "--max-len",
            "4",
            "--duv-mode",
            "reg",
            "--out-midi",
            str(midi_out),
        ]
    )
    pm = pretty_midi.PrettyMIDI(str(midi_out))
    assert pm.instruments and pm.instruments[0].notes
