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


def test_choose_preset_table() -> None:
    """
    ToneShaper がデフォルトのプリセット・ルールを保持している場合の
    マッチング動作を確認。
    """
    shaper = ToneShaper()

    # avg_velocity が 50 / intensity "low" → clean
    assert shaper.choose_preset(None, "low", 50.0) == "clean"
    # avg_velocity が 70 / intensity "medium" → drive
    assert shaper.choose_preset(None, "medium", 70.0) == "drive"
    # avg_velocity が 90 / intensity "high" → fuzz
    assert shaper.choose_preset(None, "high", 90.0) == "fuzz"
