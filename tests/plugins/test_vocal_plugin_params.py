import importlib

import pytest


@pytest.mark.plugin
def test_vocal_companion_defaults() -> None:
    mod = importlib.import_module("plugins.vocal_companion_stub")
    params = mod.get_default_parameters()
    assert params["backend"] in {"synthv", "vocaloid", "onnx"}
    assert params["model_path"] == ""
    assert params["enable_articulation"] is True
    assert 0.0 <= params["vibrato_depth"] <= 1.0
    assert 0.0 < params["vibrato_rate"] <= 10.0
