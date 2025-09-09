import json
import subprocess
from pathlib import Path

import pytest

pytest.importorskip("pretty_midi")


def test_empty_corpus_exits(tmp_path: Path) -> None:
    train = tmp_path / "train"
    valid = tmp_path / "valid"
    train.mkdir()
    valid.mkdir()
    sample = {
        "pitch": 40,
        "velocity": 80,
        "duration": 1,
        "pos": 0,
        "boundary": 0,
        "bar": 0,
        "instrument": "piano",
        "path": "piano.mid",
    }
    (train / "samples.jsonl").write_text(json.dumps(sample) + "\n")
    (valid / "samples.jsonl").write_text(json.dumps(sample) + "\n")
    out_train = tmp_path / "train.csv"
    out_valid = tmp_path / "valid.csv"
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
            str(out_train),
            "--out-valid",
            str(out_valid),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "No rows matched filters" in proc.stderr or "No rows matched filters" in proc.stdout
    assert not out_train.exists()
