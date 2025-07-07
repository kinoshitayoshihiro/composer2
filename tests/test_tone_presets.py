import pytest

from utilities.tone_shaper import ToneShaper


def test_tone_presets_ir_mapping() -> None:
    ts = ToneShaper()
    assert ts.get_ir_file("grand_clean").name == "german_grand.wav"
    assert ts.get_ir_file("grand_clean").exists()
    assert ts.get_ir_file("upright_mellow").name == "upright_1960.wav"
    assert ts.get_ir_file("ep_phase").name == "dx7_ep.wav"


def test_tone_presets_cc_values() -> None:
    ts = ToneShaper()
    ev_grand = ts.to_cc_events(amp_name="grand_clean")
    ev_up = ts.to_cc_events(amp_name="upright_mellow")
    ev_ep = ts.to_cc_events(amp_name="ep_phase")
    assert (0.0, 31, 25) in ev_grand
    assert (0.0, 31, 35) in ev_up
    assert (0.0, 31, 55) in ev_ep
    assert any(e[1] == 94 for e in ev_ep)
