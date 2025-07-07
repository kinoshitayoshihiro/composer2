import importlib

import pytest

@pytest.mark.plugin
def test_piano_companion_defaults():
    mod = importlib.import_module("plugins.piano_companion_stub")
    params = mod.get_default_parameters()
    assert params["TonePreset"] == "Default"
    assert params["Intensity"] == 1.0
    assert params["Temperature"] == 0.7
