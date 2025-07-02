from utilities.tone_shaper import ToneShaper


def test_choose_preset_drive() -> None:
    shaper = ToneShaper()
    preset = shaper.choose_preset(90.0, "high")
    assert preset == "drive"
