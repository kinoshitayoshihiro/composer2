import warnings
from pathlib import Path

import pretty_midi
from click.testing import CliRunner

from utilities import groove_sampler_ngram as gs


def _loop(p: Path) -> None:
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=0, is_drum=True)
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=36, start=0.0, end=0.1))
    pm.instruments.append(inst)
    pm.write(str(p))


def test_cli_preview_fallback(tmp_path: Path, monkeypatch) -> None:
    _loop(tmp_path / "a.mid")
    model = gs.train(tmp_path, order=1)
    gs.save(model, tmp_path / "m.pkl")
    monkeypatch.setattr(gs.shutil, "which", lambda *_: None)
    runner = CliRunner()
    with warnings.catch_warnings(record=True) as rec:
        res = runner.invoke(gs.cli, ["sample", str(tmp_path / "m.pkl"), "-l", "1", "--play"])
    assert res.exit_code == 0
    assert res.output == ""
    assert any("opened browser" in str(w.message).lower() for w in rec)
