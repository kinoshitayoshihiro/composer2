import statistics
from pathlib import Path
import pytest
from utilities.tempo_curve import TempoCurve
from utilities.velocity_smoother import EMASmoother
from utilities.timing_utils import _combine_timing


def test_tempo_interp_midpoint(tmp_path: Path) -> None:
    data = [
        {"beat": 0, "bpm": 120},
        {"beat": 32, "bpm": 108},
        {"beat": 64, "bpm": 128},
    ]
    p = tmp_path / "curve.json"
    p.write_text(str(data).replace("'", '"'), encoding="utf-8")
    curve = TempoCurve.from_json(p)
    assert curve.bpm_at(48) == pytest.approx(118.0)


def test_ema_smoothing_reduces_variance() -> None:
    vals = [120 if i % 2 == 0 else 30 for i in range(16)]
    smoother = EMASmoother()
    out = [smoother.smooth(v) for v in vals]
    assert statistics.pstdev(out) < statistics.pstdev(vals) * 0.6


def test_push_pull_swing_no_negative() -> None:
    bpm = 100
    res = _combine_timing(
        0.5,
        1.0,
        swing_ratio=0.6,
        swing_type="eighth",
        push_pull_curve=[10, -15, 5, 0],
        tempo_bpm=bpm,
        return_vel=True,
    )
    assert not res.offset_ql < 0
    assert not res.offset_ql != res.offset_ql
