import logging
from pathlib import Path

import pretty_midi
import pytest
from click.testing import CliRunner

from utilities import cli_playback
from utilities import groove_sampler_ngram as gs


def _make_loop(path: Path) -> None:
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=0, is_drum=True)
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=36, start=0.0, end=0.1))
    pm.instruments.append(inst)
    pm.write(str(path))


def test_cli_playback_fallback(tmp_path: Path, monkeypatch, caplog: pytest.LogCaptureFixture) -> None:
    _make_loop(tmp_path / "a.mid")
    model = gs.train(tmp_path, order=1)
    gs.save(model, tmp_path / "m.pkl")
    monkeypatch.setattr(cli_playback, "find_player", lambda: None)
    runner = CliRunner()
    with caplog.at_level(logging.WARNING):
        res = runner.invoke(gs.cli, ["sample", str(tmp_path / "m.pkl"), "-l", "1", "--play"])
    assert res.exit_code == 0
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING and "no MIDI player found" in r.getMessage()]
    assert len(warnings) == 1
