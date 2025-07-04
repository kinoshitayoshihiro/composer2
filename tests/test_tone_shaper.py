from utilities.tone_shaper import ToneShaper


def test_choose_preset_drive() -> None:
    shaper = ToneShaper({"drive": {"amp": 90}})
    preset = shaper.choose_preset(None, "high", 90.0)
    assert preset == "drive"
