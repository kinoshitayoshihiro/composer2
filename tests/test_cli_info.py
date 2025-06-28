from pathlib import Path
import json
import pretty_midi
from click.testing import CliRunner
from utilities import groove_sampler_ngram as gs
from modular_composer import cli


def _mk_loop(path: Path) -> None:
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=0, is_drum=True)
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=36, start=0.0, end=0.1))
    pm.instruments.append(inst)
    pm.write(str(path))


def test_root_cli_info(tmp_path: Path) -> None:
    _mk_loop(tmp_path / "a.mid")
    model = gs.train(tmp_path, order=1)
    gs.save(model, tmp_path / "m.pkl")
    runner = CliRunner()
    res = runner.invoke(cli.cli, ["groove", "info", str(tmp_path / "m.pkl"), "--json"])
    assert res.exit_code == 0
    data = json.loads(res.output)
    assert set(data.keys()) >= {"order", "contexts"}

