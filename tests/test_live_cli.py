from pathlib import Path

import pytest
from click.testing import CliRunner

from modular_composer import cli
from utilities import groove_sampler_ngram, live_player


@pytest.mark.stretch
def test_live_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dummy = tmp_path / "m.pkl"
    dummy.write_text("dummy")

    monkeypatch.setattr(groove_sampler_ngram, "load", lambda p: None)
    monkeypatch.setattr(
        groove_sampler_ngram,
        "_generate_bar",
        lambda hist, model, rng=None: ([{"instrument": "kick", "offset": 0.0}], []),
    )

    called = {
        "val": False,
    }

    def fake_play(sampler, bpm: float) -> None:
        called["val"] = True

    monkeypatch.setattr(live_player, "play_live", fake_play)

    runner = CliRunner()
    res = runner.invoke(
        cli.cli,
        ["live", str(dummy), "--backend", "ngram", "--bpm", "100"],
    )
    assert res.exit_code == 0
    assert called["val"]
