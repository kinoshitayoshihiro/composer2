import json
import subprocess
from pathlib import Path

import pytest

pretty_midi = pytest.importorskip("pretty_midi")


def make_midi(path: Path, program: int, *, is_drum: bool = False, name: str = "") -> None:
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=program, is_drum=is_drum, name=name)
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=0.5))
    pm.instruments.append(inst)
    pm.write(str(path))


def test_instrument_hints(tmp_path: Path) -> None:
    bass = tmp_path / "bass.mid"
    drum = tmp_path / "drum.mid"
    make_midi(bass, 33, name="Bass")
    make_midi(drum, 0, is_drum=True, name="Drum")
    out = tmp_path / "corpus"
    subprocess.run(
        [
            "python",
            "-m",
            "tools.prepare_transformer_corpus",
            "--in",
            str(tmp_path),
            "--out",
            str(out),
            "--bars-per-sample",
            "1",
            "--min-notes",
            "1",
            "--split",
            "1",
            "0",
            "0",
        ],
        check=True,
    )
    data = [json.loads(l) for l in (out / "train.jsonl").read_text().splitlines()]
    metas = [d["meta"] for d in data]
    bass_meta = next(m for m in metas if m["program"] == 33)
    assert bass_meta["instrument"] == "bass"
    assert bass_meta["is_drum"] is False
    drum_meta = next(m for m in metas if m["is_drum"])
    assert drum_meta["instrument"] == "drums"
    assert "track_name" in bass_meta and "path" in bass_meta
