import importlib

import pytest


@pytest.mark.plugin
def test_stub() -> None:
    mod = importlib.import_module("plugins.modcompose_stub")
    events = mod.generateBar({})
    assert isinstance(events, list)
    assert events
