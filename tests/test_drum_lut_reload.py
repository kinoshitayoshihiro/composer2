import logging
from pathlib import Path

import pytest

from generator.drum_generator import DEFAULT_FILL_DENSITY_LUT, DrumGenerator


def _make_yaml(path: Path, hi: float) -> None:
    path.write_text(
        """
        drum:
          fill_density_lut:
            0.0: 0.0
            1.0: {hi}
        """.format(
            hi=hi
        )
    )


def test_reload_lut_success(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    p = tmp_path / "lut.yml"
    _make_yaml(p, 0.5)
    gen = DrumGenerator(global_settings={"tempo_bpm": 120}, lut_path=p, main_cfg={})
    assert gen._calc_fill_density(1.0) == pytest.approx(0.5)
    _make_yaml(p, 0.9)
    with caplog.at_level(logging.INFO):
        ok = gen.reload_lut()
    assert ok
    assert gen._calc_fill_density(1.0) == pytest.approx(0.9)
    assert any("Reloaded" in r.message for r in caplog.records)


def test_reload_lut_failure(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    p = tmp_path / "lut.yml"
    p.write_text("drum: {}")
    gen = DrumGenerator(global_settings={"tempo_bpm": 120}, lut_path=p, main_cfg={})
    orig = gen.fill_density_lut.copy()
    with caplog.at_level(logging.WARNING):
        ok = gen.reload_lut()
    assert not ok
    assert gen.fill_density_lut == orig
    assert any("Reload failed" in r.message for r in caplog.records)


def test_compose_auto_reload(tmp_path: Path) -> None:
    p = tmp_path / "lut.yml"
    _make_yaml(p, 0.2)
    gen = DrumGenerator(global_settings={"tempo_bpm": 120}, lut_path=p, main_cfg={})
    assert gen.fill_density(1.0) == pytest.approx(0.2)
    _make_yaml(p, 0.7)
    section = {"q_length": 4.0, "musical_intent": {}}
    gen.compose(section_data=section)
    assert gen.fill_density(1.0) == pytest.approx(0.7)
