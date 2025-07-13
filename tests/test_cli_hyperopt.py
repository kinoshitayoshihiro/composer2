import json
import sys
from types import SimpleNamespace
import builtins

from click.testing import CliRunner

import modular_composer.cli as cli


class DummyStudy:
    def __init__(self, best):
        self.best_params = best

    def optimize(self, func, n_trials):
        pass


def test_cli_hyperopt(monkeypatch):
    best = {"temperature": 0.9}
    fake_optuna = SimpleNamespace(create_study=lambda direction="maximize": DummyStudy(best), Trial=None)
    monkeypatch.setitem(sys.modules, "optuna", fake_optuna)
    monkeypatch.setattr(cli.groove_sampler_ngram, "load", lambda p: object())
    monkeypatch.setattr(cli.groove_sampler_ngram, "sample", lambda mdl, bars, temperature: [{"velocity": 0}])
    runner = CliRunner()
    res = runner.invoke(cli.cli, ["hyperopt", "dummy.pkl", "--trials", "2"])
    assert res.exit_code == 0
    assert json.loads(res.output) == best


def test_hyperopt_skip(monkeypatch):
    orig_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "optuna":
            raise ImportError("no optuna")
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    runner = CliRunner()
    res = runner.invoke(
        cli.cli,
        ["hyperopt", "dummy.pkl", "--skip-if-no-optuna"],
    )
    assert res.exit_code == 0
    assert res.output == ""


def test_hyperopt_in_help():
    runner = CliRunner()
    res = runner.invoke(cli.cli, ["--help"])
    assert res.exit_code == 0
    assert "hyperopt" in res.output
