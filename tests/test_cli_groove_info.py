from pathlib import Path

import pretty_midi
from click.testing import CliRunner

from utilities import groove_sampler_ngram as gs


def _mk_loop(p: Path) -> None:
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0, is_drum=True)
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=36, start=0.0, end=0.1))
    pm.instruments.append(inst)
    pm.write(str(p))


def test_info_cmd(tmp_path: Path) -> None:
    _mk_loop(tmp_path / "a.mid")
    model = gs.train(tmp_path, order=1)
    gs.save(model, tmp_path / "m.pkl")
    runner = CliRunner()
    res = runner.invoke(gs.cli, ["info", str(tmp_path / "m.pkl")])
    assert res.exit_code == 0
    assert "order:" in res.output
