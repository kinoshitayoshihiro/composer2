from utilities.tone_shaper import ToneShaper


def test_unknown_style_intensity_returns_default():
    ts = ToneShaper()
    preset = ts.choose_preset(style="unknown", intensity="mystery", avg_velocity=60)
    assert preset == ts.default_preset
