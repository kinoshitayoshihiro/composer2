import json
import sys
import subprocess
from pathlib import Path


def test_validate_only(tmp_path: Path) -> None:
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
        "instrument": "foo",
    }
    (train_dir / "samples.jsonl").write_text(json.dumps(sample) + "\n")
    sample["instrument"] = "bar"
    (valid_dir / "samples.jsonl").write_text(json.dumps(sample) + "\n")
    tag_vocab = {"instrument": ["foo"]}
    vocab_path = tmp_path / "tag_vocab.json"
    vocab_path.write_text(json.dumps(tag_vocab))
    out_train = tmp_path / "train.csv"
    cmd = [
        sys.executable,
        "-m",
        "tools.corpus_to_phrase_csv",
        "--from-corpus",
        str(tmp_path),
        "--out-train",
        str(out_train),
        "--out-valid",
        str(tmp_path / "valid.csv"),
        "--validate-only",
        "--tag-vocab-in",
        str(vocab_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1])
    assert res.returncode == 0
    assert "instrument" in res.stdout
    assert "bar" in res.stdout
    assert not out_train.exists()
