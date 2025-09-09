import importlib
import sys
from pathlib import Path

from click.testing import CliRunner
import pytest

import modular_composer.cli as cli


def test_cli_no_rnn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    loops = tmp_path / "loops.json"
    loops.write_text("{}")
    monkeypatch.setitem(sys.modules, "pytorch_lightning", None)
    importlib.reload(cli)
    runner = CliRunner()
    res = runner.invoke(cli.cli, ["rnn", "train", str(loops)])
    assert res.exit_code == 1
    assert "Install extras: rnn" in res.output
