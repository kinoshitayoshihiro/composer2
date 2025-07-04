from utilities.tone_shaper import ToneShaper


# ----------------------------------------------------------------------
# ToneShaper - preset–selection tests
#  Both the explicit-table variant（旧 codex ブランチ）と
#  デフォルト・テーブル variant（旧 main ブランチ）を共存させる。
#  現在の ToneShaper.choose_preset の
#   シグネチャは (amp_hint: str | None,
#                intensity: str | None,
#                avg_velocity: float | None) -> str
# ----------------------------------------------------------------------

def test_choose_preset_drive() -> None:
    """
    プリセット・テーブルをコンストラクタで与えた場合に
    intensity と avg_velocity でマッチング出来ることを確認。
    """
    shaper = ToneShaper({"drive": {"amp": 90}})
    preset = shaper.choose_preset(None, "high", 90.0)
    assert preset == "drive"


import pytest


@pytest.mark.parametrize(
    "intensity,vel,expected",
    [
        ("low", 50.0, "clean"),
        ("low", 80.0, "crunch"),
        ("medium", 50.0, "crunch"),
        ("medium", 80.0, "drive"),
        ("high", 50.0, "drive"),
        ("high", 90.0, "fuzz"),
    ],
)
def test_choose_preset_table(intensity: str, vel: float, expected: str) -> None:
    """PRESET_TABLE mapping matrix."""
    shaper = ToneShaper({
        "clean": {"amp": 0},
        "crunch": {"amp": 32},
        "drive": {"amp": 64},
        "fuzz": {"amp": 96},
    })
    assert shaper.choose_preset(None, intensity, vel) == expected
