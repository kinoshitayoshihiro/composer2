import pytest

from utilities.tone_shaper import PRESET_LIBRARY, ToneShaper

# ----------------------------------------------------------------------
# ToneShaper - preset-selection / CC-emit tests
#   choose_preset(amp_hint=None, intensity="med", avg_velocity=64.0) -> str
# ----------------------------------------------------------------------



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
    assert shaper.choose_preset(intensity=intensity, avg_velocity=vel) == expected


# ──────────────────────────────────────────────────────────────
# Fallback 動作
# ──────────────────────────────────────────────────────────────
def test_choose_preset_fallback() -> None:
    shaper = ToneShaper({"clean": {"amp": 20}})
    # amp_hint が unknown → default へフォールバック
    assert (
        shaper.choose_preset(amp_hint="unknown", intensity="low", avg_velocity=50.0)
        == shaper.default_preset
    )


# ──────────────────────────────────────────────────────────────
# CC イベント生成：すべての CC が含まれるか
# ──────────────────────────────────────────────────────────────
def test_to_cc_events_all_cc() -> None:
    shaper = ToneShaper({"clean": {"amp": 20}})
    shaper.choose_preset(intensity="low", avg_velocity=50.0)
    events = shaper.to_cc_events(as_dict=True)
    ccs = {e["cc"] for e in events}
    assert {31, 91, 93, 94}.issubset(ccs)


# ──────────────────────────────────────────────────────────────
# Intensity によるエフェクト量スケール確認（例：Reverb CC91）
# ──────────────────────────────────────────────────────────────
def test_intensity_scaling() -> None:
    shaper = ToneShaper({"clean": {"amp": 20, "reverb": 40}})

    shaper.choose_preset(intensity="low", avg_velocity=50.0)
    low_rev = next(v for _, c, v in shaper.to_cc_events(as_dict=False) if c == 91)

    shaper.choose_preset(intensity="high", avg_velocity=90.0)
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


def test_preset_map_no_duplicates() -> None:
    ts = ToneShaper({"drive": {"amp": 80}})
    ts.choose_preset(intensity="medium", avg_velocity=70.0)
    keys = list(ts.preset_map.keys())
    assert len(keys) == len(set(keys))
    assert "drive_default" in ts.preset_map


def test_get_ir_file_fallback(tmp_path):
    ts = ToneShaper({"jam": {"amp": 60}})
    ts.ir_map["jam"] = tmp_path / "missing.wav"
    clean = tmp_path / "clean.wav"
    clean.write_text("x")
    PRESET_LIBRARY["clean"]["ir_file"] = str(clean)
    try:
        ts._selected = "jam"
        ir = ts.get_ir_file(fallback_ok=True)
        assert ir == clean
    finally:
        PRESET_LIBRARY["clean"]["ir_file"] = "data/ir/clean.wav"
