import csv
import subprocess
from pathlib import Path

import pytest

pretty_midi = pytest.importorskip("pretty_midi")
yaml = pytest.importorskip("yaml")


def make_midi(path: Path, pitches: list[int], starts: list[float] | None = None) -> None:
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0)
    t = 0.0
    for i, p in enumerate(pitches):
        start = starts[i] if starts else t
        inst.notes.append(
            pretty_midi.Note(velocity=100, pitch=p, start=start, end=start + 0.25)
        )
        t = start + 0.25
    pm.instruments.append(inst)
    pm.write(str(path))


def test_corpus_to_csv(tmp_path: Path) -> None:
    m1 = tmp_path / "a.mid"
    m2 = tmp_path / "b.mid"
    make_midi(m1, [60, 62], starts=[0.0, 0.25])
    make_midi(m2, [65])
    out_train = tmp_path / "train.csv"
    out_valid = tmp_path / "valid.csv"
    tags = {"a.mid": {"section": [[0.0, "A"], [0.2, "B"]]}}
    (tmp_path / "tags.yaml").write_text(yaml.safe_dump(tags))
    subprocess.run(
        [
            "python",
            "-m",
            "tools.corpus_to_phrase_csv",
            "--in",
            str(tmp_path),
            "--out-train",
            str(out_train),
            "--out-valid",
            str(out_valid),
            "--boundary-on-section-change",
        ],
        check=True,
    )
    for csv_path in (out_train, out_valid):
        with csv_path.open() as f:
            rows = list(csv.DictReader(f))
        assert rows, f"{csv_path} empty"
        header = rows[0].keys()
        required = {
            "pitch",
            "velocity",
            "duration",
            "pos",
            "boundary",
            "bar",
            "instrument",
            "program",
        }
        assert required.issubset(header)
    with out_train.open() as f:
        rows = list(csv.DictReader(f))
    assert rows[1]["boundary"] == "1"
    assert {row["program"] for row in rows} == {"0"}


def test_from_corpus(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    valid_dir = tmp_path / "valid"
    train_dir.mkdir()
    valid_dir.mkdir()
    import json

    train_sample = {
        "pitch": 60,
        "velocity": 100,
        "duration": 1,
        "pos": 0,
        "boundary": 1,
        "bar": 0,
        "instrument": "",
    }
    valid_sample = train_sample | {"pitch": 62}
    (train_dir / "samples.jsonl").write_text(json.dumps(train_sample) + "\n")
    (valid_dir / "samples.jsonl").write_text(json.dumps(valid_sample) + "\n")
    out_train = tmp_path / "train.csv"
    out_valid = tmp_path / "valid.csv"
    subprocess.run(
        [
            "python",
            "-m",
            "tools.corpus_to_phrase_csv",
            "--from-corpus",
            str(tmp_path),
            "--out-train",
            str(out_train),
            "--out-valid",
            str(out_valid),
        ],
        check=True,
    )
    for csv_path in (out_train, out_valid):
        with csv_path.open() as f:
            rows = list(csv.DictReader(f))
        assert rows
        assert "program" in rows[0]
        assert rows[0]["program"] == "-1"


def test_unknown_instrument_fallback(tmp_path: Path) -> None:
    import json

    train_dir = tmp_path / "train"
    valid_dir = tmp_path / "valid"
    train_dir.mkdir()
    valid_dir.mkdir()
    sample = {
        "pitch": 40,
        "velocity": 80,
        "duration": 1,
        "pos": 0,
        "boundary": 0,
        "bar": 0,
        "instrument": "unknown",
        "path": "My_Remix_Bass.mid",
    }
    (train_dir / "samples.jsonl").write_text(json.dumps(sample) + "\n")
    (valid_dir / "samples.jsonl").write_text(json.dumps(sample) + "\n")
    out_train = tmp_path / "train.csv"
    out_valid = tmp_path / "valid.csv"
    subprocess.run(
        [
            "python",
            "-m",
            "tools.corpus_to_phrase_csv",
            "--from-corpus",
            str(tmp_path),
            "--instrument",
            "bass",
            "--pitch-range",
            "28",
            "60",
            "--out-train",
            str(out_train),
            "--out-valid",
            str(out_valid),
        ],
        check=True,
    )
    with out_train.open() as f:
        rows = list(csv.DictReader(f))
    assert rows


def test_dry_run(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    valid_dir = tmp_path / "valid"
    train_dir.mkdir()
    valid_dir.mkdir()
    import json

    sample = {
        "pitch": 60,
        "velocity": 100,
        "duration": 1,
        "pos": 0,
        "boundary": 1,
        "bar": 0,
        "instrument": "",
    }
    (train_dir / "samples.jsonl").write_text(json.dumps(sample) + "\n")
    (valid_dir / "samples.jsonl").write_text(json.dumps(sample) + "\n")
    out_train = tmp_path / "train.csv"
    subprocess.run(
        [
            "python",
            "-m",
            "tools.corpus_to_phrase_csv",
            "--from-corpus",
            str(tmp_path),
            "--out-train",
            str(out_train),
            "--out-valid",
            str(tmp_path / "valid.csv"),
            "--dry-run",
        ],
        check=True,
    )
    assert not out_train.exists()


def test_or_and_filters(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    valid_dir = tmp_path / "valid"
    train_dir.mkdir()
    valid_dir.mkdir()
    import json

    sample = {
        "pitch": 40,
        "velocity": 80,
        "duration": 1,
        "pos": 0,
        "boundary": 1,
        "bar": 0,
        "instrument": "piano",
        "path": "foo.mid",
    }
    js = json.dumps(sample) + "\n"
    (train_dir / "samples.jsonl").write_text(js)
    (valid_dir / "samples.jsonl").write_text(js)
    out_train = tmp_path / "train.csv"
    out_valid = tmp_path / "valid.csv"
    subprocess.run(
        [
            "python",
            "-m",
            "tools.corpus_to_phrase_csv",
            "--from-corpus",
            str(tmp_path),
            "--instrument",
            "bass",
            "--pitch-range",
            "38",
            "41",
            "--out-train",
            str(out_train),
            "--out-valid",
            str(out_valid),
        ],
        check=True,
    )
    assert out_train.read_text().strip().splitlines()[1]
    res = subprocess.run(
        [
            "python",
            "-m",
            "tools.corpus_to_phrase_csv",
            "--from-corpus",
            str(tmp_path),
            "--instrument",
            "bass",
            "--pitch-range",
            "38",
            "41",
            "--strict-all",
            "--out-train",
            str(tmp_path / "train2.csv"),
            "--out-valid",
            str(tmp_path / "valid2.csv"),
        ],
        check=False,
    )
    assert res.returncode != 0


def test_min_notes_threshold(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    valid_dir = tmp_path / "valid"
    train_dir.mkdir()
    valid_dir.mkdir()
    import json

    sample = {
        "pitch": 60,
        "velocity": 90,
        "duration": 1,
        "pos": 0,
        "boundary": 1,
        "bar": 0,
        "instrument": "bass",
        "notes": [1, 2, 3, 4],
    }
    js = json.dumps(sample) + "\n"
    (train_dir / "samples.jsonl").write_text(js)
    (valid_dir / "samples.jsonl").write_text(js)
    res = subprocess.run(
        [
            "python",
            "-m",
            "tools.corpus_to_phrase_csv",
            "--from-corpus",
            str(tmp_path),
            "--min-notes-per-sample",
            "8",
            "--out-train",
            str(tmp_path / "train.csv"),
            "--out-valid",
            str(tmp_path / "valid.csv"),
        ],
        check=False,
    )
    assert res.returncode != 0


def test_dry_run_stats_json(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    valid_dir = tmp_path / "valid"
    train_dir.mkdir()
    valid_dir.mkdir()
    import json

    sample = {
        "pitch": 60,
        "velocity": 100,
        "duration": 1.0,
        "pos": 0,
        "boundary": 1,
        "bar": 0,
        "instrument": "",
    }
    js = json.dumps(sample) + "\n"
    (train_dir / "samples.jsonl").write_text(js)
    (valid_dir / "samples.jsonl").write_text(js)
    stats_path = tmp_path / "stats.json"
    res = subprocess.run(
        [
            "python",
            "-m",
            "tools.corpus_to_phrase_csv",
            "--from-corpus",
            str(tmp_path),
            "--out-train",
            str(tmp_path / "train.csv"),
            "--out-valid",
            str(tmp_path / "valid.csv"),
            "--dry-run",
            "--stats-json",
            str(stats_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "train kept" in res.stdout or "train" in res.stderr
    data = json.loads(stats_path.read_text())
    assert "train" in data and "pitch_hist" in data["train"]
