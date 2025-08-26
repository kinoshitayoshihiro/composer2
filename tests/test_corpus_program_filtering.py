import csv
import json
import subprocess
from pathlib import Path


def test_empty_extraction_returns_nonzero(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    valid_dir = tmp_path / "valid"
    train_dir.mkdir()
    valid_dir.mkdir()
    sample = {
        "pitch": 60,
        "velocity": 100,
        "duration": 1,
        "pos": 0,
        "boundary": 1,
        "bar": 0,
        "instrument": "piano",
    }
    (train_dir / "samples.jsonl").write_text(json.dumps(sample) + "\n")
    (valid_dir / "samples.jsonl").write_text(json.dumps(sample) + "\n")
    stats_path = tmp_path / "stats.json"
    proc = subprocess.run(
        [
            "python",
            "-m",
            "tools.corpus_to_phrase_csv",
            "--from-corpus",
            str(tmp_path),
            "--instrument",
            "bass",
            "--out-train",
            str(tmp_path / "train.csv"),
            "--out-valid",
            str(tmp_path / "valid.csv"),
            "--stats-json",
            str(stats_path),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 1
    data = json.loads(stats_path.read_text())
    assert data["train"]["kept"] == 0
    assert "removed_by_name" in data["train"]
    assert data["valid"]["kept"] == 0


def test_program_filtering(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    valid_dir = tmp_path / "valid"
    train_dir.mkdir()
    valid_dir.mkdir()
    sample_bass = {
        "pitch": 40,
        "velocity": 80,
        "duration": 1,
        "pos": 0,
        "boundary": 1,
        "bar": 0,
        "meta": {"instrument": "bass", "program": 32},
    }
    sample_piano = {
        "pitch": 60,
        "velocity": 80,
        "duration": 1,
        "pos": 0,
        "boundary": 1,
        "bar": 0,
        "meta": {"instrument": "piano", "program": 0},
    }
    (train_dir / "samples.jsonl").write_text(
        json.dumps(sample_bass) + "\n" + json.dumps(sample_piano) + "\n"
    )
    (valid_dir / "samples.jsonl").write_text(json.dumps(sample_bass) + "\n")
    out_train = tmp_path / "train.csv"
    out_valid = tmp_path / "valid.csv"
    subprocess.run(
        [
            "python",
            "-m",
            "tools.corpus_to_phrase_csv",
            "--from-corpus",
            str(tmp_path),
            "--include-programs",
            "32",
            "--out-train",
            str(out_train),
            "--out-valid",
            str(out_valid),
        ],
        check=True,
    )
    with out_train.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["instrument"] == "bass"

