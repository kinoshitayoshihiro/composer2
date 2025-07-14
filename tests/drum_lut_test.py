import logging
from pathlib import Path

import pytest

from generator.drum_generator import DEFAULT_FILL_DENSITY_LUT, DrumGenerator


@pytest.mark.parametrize(
    "yaml_text,expect_default",
    [
        (
            """
            drum:
              fill_density_lut:
                0.2: 0.1
                0.5: 0.3
            """,
            False,
        ),
        (
            """
            drum:
              fill_density_lut:
                0.0: 0.05
                0.5: 0.30
                0.2: 0.10
                1.0: 0.65
            """,
            False,
        ),
        (
            """
            drum:
              fill_density_lut:
                0.0: bad
                1.0: 0.65
            """,
            True,
        ),
    ],
)
def test_lut_loading_cases(
    tmp_path: Path,
    yaml_text: str,
    expect_default: bool,
    caplog: pytest.LogCaptureFixture,
):
    path = tmp_path / "lut.yml"
    path.write_text(yaml_text)
    with caplog.at_level(logging.WARNING):
        gen = DrumGenerator(
            global_settings={"tempo_bpm": 120}, lut_path=path, main_cfg={}
        )
    if expect_default:
        assert gen.fill_density_lut == DEFAULT_FILL_DENSITY_LUT
        assert any(r.levelno == logging.WARNING for r in caplog.records)
    else:
        assert gen.lut_path == path
        assert not any(r.levelno >= logging.ERROR for r in caplog.records)


def test_env_var_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    env_yaml = tmp_path / "env.yml"
    env_yaml.write_text(
        """
        drum:
          fill_density_lut:
            0.0: 0.0
            1.0: 0.5
        """
    )
    monkeypatch.setenv("DRUM_LUT_PATH", str(env_yaml))
    gen = DrumGenerator(global_settings={"tempo_bpm": 120}, main_cfg={})
    assert gen._calc_fill_density(1.0) == pytest.approx(0.5)
    monkeypatch.delenv("DRUM_LUT_PATH")
