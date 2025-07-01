import importlib
import sys

import pytest


def test_rnn_lazy_import(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "pytorch_lightning", None)
    import modular_composer.cli as cli
    importlib.reload(cli)
    assert hasattr(cli, "cli")
