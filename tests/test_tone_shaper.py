import pytest

from utilities.tone_shaper import ToneShaper

# ----------------------------------------------------------------------
# ToneShaper - preset-selection / CC-emit tests
#   choose_preset(amp_hint, intensity, avg_velocity) -> str
# ----------------------------------------------------------------------

def test_choose_preset_drive() -> None:
    """
    intensity=high & avg_velocity=90 だが、
    プリセットマップに "fuzz" が無い場合は default にフォールバックする。
    """
    shaper = ToneShaper({"drive": {"amp": 90}})
    preset = shaper.choose_preset(None, "high", 90.0)
    assert preset == "drive"


def test_choose_preset_table() -> None:
    """
    ToneShaper がデフォルトのプリセット・ルールを保持している場合の
    マッチング動作を確認。
    """
    shaper = ToneShaper()

    # avg_velocity が 50 / intensity "low" → clean
    assert shaper.choose_preset(None, "low", 50.0) == "clean"
    # avg_velocity が 70 / intensity "medium" → drive (存在しなければ clean)
    assert shaper.choose_preset(None, "medium", 70.0) == "drive"
    # avg_velocity が 90 / intensity "high" → fuzz (存在しなければ clean)
    assert shaper.choose_preset(None, "high", 90.0) == "fuzz"
    assert preset == "clean"        # default_fallback


# ──────────────────────────────────────────────────────────────
# PRESET_TABLE マトリクス通りの動作確認
# ──────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "intensity, vel, expected",
    [
        ("low",    50.0, "clean"),
        ("low",    80.0, "crunch"),
        ("medium", 50.0, "crunch"),
        ("medium", 80.0, "drive"),
        ("high",   50.0, "drive"),
        ("high",   90.0, "fuzz"),
    ],
)
def test_choose_preset_table(intensity: str, vel: float, expected: str) -> None:
    shaper = ToneShaper(
        {
            "clean":  {"amp": 0},
            "crunch": {"amp": 32},
            "drive":  {"amp": 64},
            "fuzz":   {"amp": 96},
        }
    )
    assert shaper.choose_preset(None, intensity, vel) == expected


# ──────────────────────────────────────────────────────────────
# Fallback 動作
# ──────────────────────────────────────────────────────────────
def test_choose_preset_fallback() -> None:
    shaper = ToneShaper({"clean": {"amp": 20}})
    # amp_hint が unknown → default へフォールバック
    assert shaper.choose_preset("unknown", "low", 50.0) == "clean"


# ──────────────────────────────────────────────────────────────
# CC イベント生成：すべての CC が含まれるか
# ──────────────────────────────────────────────────────────────
def test_to_cc_events_all_cc() -> None:
    shaper = ToneShaper({"clean": {"amp": 20}})
    shaper.choose_preset(None, "low", 50.0)            # preset を選択
    events = shaper.to_cc_events(as_dict=True)
    ccs = {e["cc"] for e in events}
    assert {31, 91, 93, 94}.issubset(ccs)


# ──────────────────────────────────────────────────────────────
# Intensity によるエフェクト量スケール確認（例：Reverb CC91）
# ──────────────────────────────────────────────────────────────
def test_intensity_scaling() -> None:
    shaper = ToneShaper({"clean": {"amp": 20, "reverb": 40}})

    shaper.choose_preset(None, "low", 50.0)
    low_rev = next(v for _, c, v in shaper.to_cc_events(as_dict=False) if c == 91)

    shaper.choose_preset(None, "high", 90.0)
    high_rev = next(v for _, c, v in shaper.to_cc_events(as_dict=False) if c == 91)

    assert high_rev > low_rev


# ──────────────────────────────────────────────────────────────
# YAML ロードのバリデーション
# ──────────────────────────────────────────────────────────────
def test_from_yaml_invalid_value(tmp_path) -> None:
    """malformed YAML では ValueError を発生させる。"""
    bad_yaml = tmp_path / "preset.yml"
    bad_yaml.write_text("presets: {bad: 200}\nir: {bad: foo.wav}")
    with pytest.raises(ValueError):
        ToneShaper.from_yaml(bad_yaml)
