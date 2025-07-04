from utilities.tone_shaper import ToneShaper


def test_choose_preset_table() -> None:
    shaper = ToneShaper()
    assert shaper.choose_preset(50.0, "low") == "clean"
    assert shaper.choose_preset(70.0, "medium") == "drive"
    assert shaper.choose_preset(90.0, "high") == "fuzz"
