import json
import subprocess
from pathlib import Path

import pytest

try:
    import pretty_midi  # type: ignore
except Exception:  # pragma: no cover
    from ._stubs import pretty_midi  # type: ignore

from utilities import audio_to_midi_batch, apply_controls
from utilities.audio_to_midi_batch import StemResult
from utilities.controls_spline import ControlCurve


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
            "--controls-args=--write-rpn-range",
        ]
    )

    assert len(calls) == 1
    assert str(routing) in calls[0]
    assert "utilities.apply_controls_cli" in calls[0]
    midi_files = list((out_dir / in_dir.name).glob("*.mid"))
    assert len(midi_files) == 1


def test_write_rpn_flag_passed(tmp_path, monkeypatch):
    in_dir = tmp_path / "wav"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    (in_dir / "a.wav").write_bytes(b"")

    def _stub(path: Path, **_):
        inst = pretty_midi.Instrument(program=0, name=path.stem)
        inst.notes.append(pretty_midi.Note(60, 60, start=0, end=1))
        return StemResult(inst, 120.0)

    captured: dict[str, object] = {}

    def fake_apply(pm, curves_by_channel, **kw):
        captured.update(kw)
        ch = pretty_midi.Instrument(program=0, name="channel0")
        ch.control_changes.append(pretty_midi.ControlChange(number=11, value=1, time=0.0))
        ch.pitch_bends.append(pretty_midi.PitchBend(pitch=100, time=0.5))
        pm.instruments.append(ch)
        return pm

    written: dict[str, pretty_midi.PrettyMIDI] = {}

    def fake_write(self, path):  # type: ignore[override]
        written["pm"] = self
        Path(path).write_bytes(b"")

    monkeypatch.setattr(audio_to_midi_batch, "_transcribe_stem", _stub)
    monkeypatch.setattr(audio_to_midi_batch, "apply_controls", fake_apply)
    monkeypatch.setattr(pretty_midi.PrettyMIDI, "write", fake_write)

    audio_to_midi_batch.main(
        [
            str(in_dir),
            str(out_dir),
            "--jobs",
            "1",
            "--emit-cc11",
            "--cc-strategy",
            "rms",
            "--write-rpn-range",
        ]
    )

    assert captured.get("write_rpn") is True
    midi_files = list((out_dir / in_dir.name).glob("*.mid"))
    assert len(midi_files) == 1
    pm_out = written.get("pm")
    assert pm_out is not None
    assert len(pm_out.instruments) == 1
    inst = pm_out.instruments[0]
    assert inst.notes
    assert inst.control_changes and inst.pitch_bends


@pytest.mark.parametrize("mode", ["add", "skip", "replace"])
def test_controls_post_bend_modes(mode):
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0, name="channel0")
    inst.notes.append(pretty_midi.Note(60, 100, start=0, end=1))
    inst.pitch_bends.append(pretty_midi.PitchBend(pitch=1000, time=0.5))
    pm.instruments.append(inst)
    bend_curve = ControlCurve([0, 1], [0, 1])
    by_ch = {0: {"bend": bend_curve}}
    ch_map = {"bend": 0}
    if mode in {"replace", "skip"}:
        has_existing = bool(inst.pitch_bends)
        if mode == "replace" and has_existing:
            inst.pitch_bends.clear()
        elif mode == "skip" and has_existing:
            ch_bend = ch_map.get("bend", 0)
            if ch_bend in by_ch and "bend" in by_ch[ch_bend]:
                del by_ch[ch_bend]["bend"]
                if not by_ch[ch_bend]:
                    del by_ch[ch_bend]
    apply_controls.apply_controls(pm, by_ch, write_rpn=False)
    bends = pm.instruments[0].pitch_bends
    if mode == "add":
        assert len(bends) > 1
    elif mode == "skip":
        assert len(bends) == 1
    else:
        assert bends and bends[0].time != pytest.approx(0.5)
