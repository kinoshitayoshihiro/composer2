from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write_samples(path: Path, rows: list[dict[str, object]]) -> None:
    text = "\n".join(json.dumps(r) for r in rows) + "\n"
    path.write_text(text)


def test_list_instruments_json(tmp_path: Path) -> None:
    train = tmp_path / "train"
    valid = tmp_path / "valid"
    train.mkdir()
    valid.mkdir()
    _write_samples(
        train / "samples.jsonl",
        [
            {"instrument": "Piano", "track_name": "Keys", "program": 0, "path": "t1.mid"},
            {"instrument": "Guitar", "track_name": "Gtr", "program": 24, "path": "t2.mid"},
        ],
    )
    _write_samples(
        valid / "samples.jsonl",
        [
            {"instrument": "Piano", "track_name": "Keys", "program": 0, "path": "v1.mid"}
        ],
    )
    cmd = [
        sys.executable,
        "-m",
        "tools.corpus_to_phrase_csv",
        "--from-corpus",
        str(tmp_path),
        "--list-instruments",
        "--json",
        "--examples-per-key",
        "1",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert res.returncode == 0
    data = json.loads(res.stdout)
    assert data["instrument"]["piano"]["count"] == 2
    assert len(data["instrument"]["piano"]["examples"]) == 1
    assert "guitar" in data["instrument"]
