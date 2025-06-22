import pytest
from pathlib import Path
from utilities.tempo_curve import TempoCurve
from music21 import meter
from hypothesis import given, strategies as st
from generator.drum_generator import _combine_timing


def test_tempo_curve_linear(tmp_path: Path) -> None:
    data = [
        {"beat": 0, "bpm": 120},
        {"beat": 32, "bpm": 105},
        {"beat": 64, "bpm": 115},
    ]
    path = tmp_path / "curve.json"
    path.write_text(str(data).replace("'", '"'), encoding="utf-8")

    curve = TempoCurve.from_json(path)
    assert curve.bpm_at(16) == pytest.approx(112.5)
    assert curve.bpm_at(48) == pytest.approx(110)
    assert curve.bpm_at(-10) == 120
    assert curve.bpm_at(80) == 115


curve4 = TempoCurve([{"beat": 0, "bpm": 60}, {"beat": 4, "bpm": 120}])
ts44 = meter.TimeSignature("4/4")


@given(st.floats(0, 4), st.floats(-50, 50))
def test_combine_timing_ms_roundtrip(off_beat: float, shift_ms: float) -> None:
    bpm = curve4.bpm_at(off_beat, ts44)
    base = 0.5
    blend = _combine_timing(
        base,
        1.0,
        swing_ratio=0.5,
        swing_type="eighth",
        push_pull_curve=[shift_ms],
        tempo_bpm=bpm,
        return_vel=True,
    )
    ms_back = (blend.offset_ql - base) * 60.0 / bpm * 1000.0
    assert ms_back == pytest.approx(shift_ms * 0.5, abs=1.0)

