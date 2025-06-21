import json
from pathlib import Path

from generator.drum_generator import _combine_timing
from utilities.tempo_utils import get_bpm_at


def test_tempo_curve_linear(tmp_path: Path) -> None:
    curve = [
        {"beat": 0, "bpm": 110},
        {"beat": 32, "bpm": 105},
        {"beat": 64, "bpm": 115},
        {"beat": 96, "bpm": 110},
    ]
    path = tmp_path / "curve.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(curve, fh)

    for beat in range(0, 128, 16):
        bpm = get_bpm_at(beat, curve)
        blend = _combine_timing(
            0.0,
            1.0,
            swing_ratio=0.5,
            swing_type="eighth",
            push_pull_curve=[20.0],
            tempo_bpm=bpm,
            max_push_ms=80.0,
            vel_range=(0.9, 1.1),
        )
        offset_ql = blend
        # 20 ms push over one beat -> convert to quarterLength (0.5 factor)
        denom = 20.0 / 1000.0 * 0.5
        est_bpm = offset_ql / denom * 60.0
        assert abs(est_bpm - bpm) <= 0.5
