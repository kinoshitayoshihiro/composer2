import json
import subprocess
from pathlib import Path

import pytest

try:
    import pretty_midi  # type: ignore
except Exception:  # pragma: no cover
    from ._stubs import pretty_midi  # type: ignore

from utilities import audio_to_midi_batch
from utilities.audio_to_midi_batch import StemResult


def _curve_files(tmp_path: Path) -> Path:
    curve = {"domain": "time", "knots": [[0, 0], [1, 1]]}
    curve_path = tmp_path / "bend.json"
    with curve_path.open("w") as fh:
        json.dump(curve, fh)
    routing = {"0": {"bend": str(curve_path)}}
    routing_path = tmp_path / "routing.json"
    with routing_path.open("w") as fh:
        json.dump(routing, fh)
    return routing_path


def test_batch_invokes_apply_controls(tmp_path, monkeypatch):
    in_dir = tmp_path / "wav"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    (in_dir / "a.wav").write_bytes(b"")
    routing = _curve_files(tmp_path)

    def _stub(
        path: Path,
        *,
        step_size=10,
        conf_threshold=0.5,
        min_dur=0.05,
        auto_tempo=True,
        **kwargs,
    ):
        inst = pretty_midi.Instrument(program=0, name=path.stem)
        return StemResult(inst, 120.0)

    calls: list[list[str]] = []

    def fake_run(cmd, check, capture_output, text):
        calls.append(cmd)

        class R:
            returncode = 0
            stdout = ""
            stderr = ""

        return R()

    monkeypatch.setattr(audio_to_midi_batch, "_transcribe_stem", _stub)
    monkeypatch.setattr(subprocess, "run", fake_run)

    audio_to_midi_batch.main(
        [
            str(in_dir),
            str(out_dir),
            "--jobs",
            "1",
            "--controls-routing",
            str(routing),
            "--controls-args=--write-rpn",
        ]
    )

    assert len(calls) == 1
    assert str(routing) in calls[0]
    midi_files = list((out_dir / in_dir.name).glob("*.mid"))
    assert len(midi_files) == 1
