import pytest
from pathlib import Path
from utilities.tempo_curve import TempoCurve


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

