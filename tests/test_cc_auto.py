import math
import subprocess
import sys

import pytest

np = pytest.importorskip("numpy", reason="optional dep")
pretty_midi = pytest.importorskip("pretty_midi", reason="optional dep")
from utilities import audio_to_midi_batch  # noqa: E402

from utilities.audio_to_midi_batch import (  # noqa: E402  # isort: skip
    _emit_pitch_bend_range,
    apply_cc_curves,
)


def test_cc11_energy_rms():
    sr = 16000
    half = sr // 2
    audio = np.concatenate([np.ones(half) * 0.1, np.ones(half) * 0.9])
    inst = pretty_midi.Instrument(program=0, name="piano")
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=1.0))
    apply_cc_curves(
        inst,
        audio=audio,
        sr=sr,
        tempo=120.0,
        cc11_strategy="energy",
        cc11_map="linear",
        cc11_smooth_ms=60.0,
        cc11_gain=1.0,
        cc11_hyst_up=3.0,
        cc11_hyst_down=3.0,
        cc11_min_dt_ms=30.0,
        cc64_mode="none",
        cc64_gap_beats=0.25,
        cc64_min_dwell_ms=80.0,
        track_name="piano",
    )
    vals = [cc.value for cc in inst.control_changes if cc.number == 11]
    assert vals[0] < vals[-1]
    assert min(vals) >= 0 and max(vals) <= 127


def test_cc11_none_no_audio():
    inst = pretty_midi.Instrument(program=0, name="piano")
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=1.0))
    apply_cc_curves(
        inst,
        audio=None,
        sr=None,
        tempo=120.0,
        cc11_strategy="none",
        cc11_map="linear",
        cc11_smooth_ms=60.0,
        cc11_gain=1.0,
        cc11_hyst_up=3.0,
        cc11_hyst_down=3.0,
        cc11_min_dt_ms=30.0,
        cc64_mode="none",
        cc64_gap_beats=0.25,
        cc64_min_dwell_ms=80.0,
        track_name="piano",
    )
    assert not any(cc.number == 11 for cc in inst.control_changes)


def test_cc64_link_short_gaps():
    inst = pretty_midi.Instrument(program=0, name="piano")
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=0.5))
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.56, end=1.0))
    apply_cc_curves(
        inst,
        audio=None,
        sr=None,
        tempo=120.0,
        cc11_strategy="none",
        cc11_map="linear",
        cc11_smooth_ms=60.0,
        cc11_gain=1.0,
        cc11_hyst_up=3.0,
        cc11_hyst_down=3.0,
        cc11_min_dt_ms=30.0,
        cc64_mode="heuristic",
        cc64_gap_beats=0.25,
        cc64_min_dwell_ms=80.0,
        track_name="piano",
    )
    events = [(cc.time, cc.value) for cc in inst.control_changes if cc.number == 64]
    assert events[0][1] == 127 and math.isclose(events[0][0], 0.5, abs_tol=0.01)
    assert events[1][1] == 0 and events[1][0] > events[0][0]


def test_cc11_hyst_min_dt_reduce_events():
    sr = 1000
    hop = sr // 50
    audio = np.zeros(sr)
    for i in range(50):
        audio[i * hop : (i + 1) * hop] = i / 50
    inst = pretty_midi.Instrument(program=0, name="piano")
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=1.0))

    def count(h_up, h_down, min_dt):
        inst.control_changes.clear()
        apply_cc_curves(
            inst,
            audio=audio,
            sr=sr,
            tempo=120.0,
            cc11_strategy="energy",
            cc11_map="linear",
            cc11_smooth_ms=0.0,
            cc11_gain=1.0,
            cc11_hyst_up=h_up,
            cc11_hyst_down=h_down,
            cc11_min_dt_ms=min_dt,
            cc64_mode="none",
            cc64_gap_beats=0.25,
            cc64_min_dwell_ms=80.0,
            track_name="piano",
        )
        return len([cc for cc in inst.control_changes if cc.number == 11])

    n1 = count(3, 3, 30)
    n2 = count(10, 10, 30)
    n3 = count(10, 10, 80)
    assert n2 <= n1 and n3 <= n2


def test_cc64_gap_beats_reduction():
    def events_for(gap_beats: float) -> int:
        inst = pretty_midi.Instrument(program=0, name="piano")
        inst.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=0.5))
        inst.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.56, end=1.0))
        apply_cc_curves(
            inst,
            audio=None,
            sr=None,
            tempo=120.0,
            cc11_strategy="none",
            cc11_map="linear",
            cc11_smooth_ms=60.0,
            cc11_gain=1.0,
            cc11_hyst_up=3.0,
            cc11_hyst_down=3.0,
            cc11_min_dt_ms=30.0,
            cc64_mode="heuristic",
            cc64_gap_beats=gap_beats,
            cc64_min_dwell_ms=80.0,
            track_name="piano",
        )
        return len([cc for cc in inst.control_changes if cc.number == 64])

    n1 = events_for(0.25)
    n2 = events_for(0.05)
    assert n2 < n1


def test_base_generator_controls():
    pytest.importorskip("music21", reason="music21 missing")
    from music21 import note, stream

    from generator.base_part_generator import BasePartGenerator
    from utilities.cc_tools import finalize_cc_events

    class _DummyGen(BasePartGenerator):
        def _render_part(self, *args, **kwargs):  # pragma: no cover - not used
            return stream.Part()

    part = stream.Part()
    part.append(note.Note("C4", quarterLength=0.5))
    part.append(note.Note("E4", quarterLength=0.5, offset=0.62))
    gen = _DummyGen(
        default_instrument="Acoustic Piano",
        controls={
            "enable_cc11": True,
            "cc11_shape": "pad",
            "cc11_depth": 1.0,
            "enable_sustain": True,
            "sustain_mode": "heuristic",
        },
    )
    gen._apply_controls(part)
    finalize_cc_events(part)
    times = [e["time"] for e in part.extra_cc]
    assert times == sorted(times)
    for e in part.extra_cc:
        assert 0 <= e["val"] <= 127
    nums = {e["cc"] for e in part.extra_cc}
    assert {11, 64}.issubset(nums)


def test_bend_integer_range_sets_lsb_zero():
    inst = pretty_midi.Instrument(program=0)
    _emit_pitch_bend_range(inst, 2.5, integer_only=True)
    vals = [cc.value for cc in inst.control_changes if cc.number == 38]
    assert vals == [0]


def test_cli_help_flags():
    result = subprocess.run(
        [sys.executable, "-m", "utilities.audio_to_midi_batch", "-h"],
        capture_output=True,
        text=True,
        check=True,
    )
    out = result.stdout
    for flag in [
        "--cc11-strategy",
        "--cc11-map",
        "--cc11-smooth-ms",
        "--cc64-mode",
        "--bend-integer-range",
    ]:
        assert flag in out


def test_tempo_lock_log_summary(monkeypatch, tmp_path, caplog):
    src = tmp_path / "in"
    src.mkdir()
    for name in ["drums", "piano"]:
        (src / f"{name}.wav").write_text("")

    class _Res:
        def __init__(self, name: str, tempo: float):
            self.instrument = pretty_midi.Instrument(program=0, name=name)
            self.tempo = tempo

    tempos = {"drums": 120.0, "piano": 60.0}

    def fake_transcribe(path, **kwargs):
        return _Res(path.stem, tempos[path.stem])

    monkeypatch.setattr(audio_to_midi_batch, "_transcribe_stem", fake_transcribe)

    dst = tmp_path / "out1"
    caplog.set_level("INFO")
    audio_to_midi_batch.convert_directory(
        src,
        dst,
        merge=True,
        tempo_lock="anchor",
        tempo_anchor_pattern="drums",
    )
    summary = [r.message for r in caplog.records if r.message.startswith("Tempo-lock")][-1]
    assert "candidates=2" in summary and "fold" not in summary

    caplog.clear()
    dst2 = tmp_path / "out2"
    audio_to_midi_batch.convert_directory(
        src,
        dst2,
        merge=True,
        tempo_lock="anchor",
        tempo_anchor_pattern="drums",
        tempo_fold_halves=True,
    )
    summary = [r.message for r in caplog.records if r.message.startswith("Tempo-lock")][-1]
    assert "candidates=2" in summary and "fold" in summary
