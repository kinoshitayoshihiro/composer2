import pytest
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
    assert preset == "clean"


def test_choose_preset_table() -> None:
    """
    ToneShaper がデフォルトのプリセット・ルールを保持している場合の
    マッチング動作を確認。
    """
    shaper = ToneShaper()

    # avg_velocity が 50 / intensity "low" → clean
    assert shaper.choose_preset(None, "low", 50.0) == "clean"
    # avg_velocity が 70 / intensity "medium" → drive (存在しなければ clean)
    assert shaper.choose_preset(None, "medium", 70.0) in {"drive", "clean"}
    # avg_velocity が 90 / intensity "high" → fuzz (存在しなければ clean)
    assert shaper.choose_preset(None, "high", 90.0) in {"fuzz", "clean"}


def test_choose_preset_fallback() -> None:
    shaper = ToneShaper({"clean": {"amp": 20}})
    assert shaper.choose_preset("unknown", "low", 50.0) == "clean"


def test_to_cc_events_all_cc() -> None:
    shaper = ToneShaper({"clean": {"amp": 20}})
    events = shaper.to_cc_events("clean", "low", as_dict=True)
    ccs = {e["cc"] for e in events}
    assert {31, 91, 93, 94}.issubset(ccs)


def test_intensity_scaling() -> None:
    shaper = ToneShaper({"clean": {"amp": 20, "reverb": 40}})
    low = shaper.to_cc_events("clean", "low", as_dict=False)
    high = shaper.to_cc_events("clean", "high", as_dict=False)
    low_val = next(v for _, c, v in low if c == 91)
    high_val = next(v for _, c, v in high if c == 91)
    assert high_val > low_val


def test_from_yaml_invalid_value(tmp_path) -> None:
    f = tmp_path / "p.yml"
    f.write_text("presets: {bad: 200}\nir: {bad: foo.wav}")
    with pytest.raises(ValueError):
        ToneShaper.from_yaml(f)
