from pathlib import Path

import pretty_midi
import pytest
from click.testing import CliRunner

from modular_composer import cli


def _mk_loop(path: Path) -> None:
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=0, is_drum=True)
    for i in range(8):
        start = i * 0.5
        inst.notes.append(pretty_midi.Note(velocity=80, pitch=36, start=start, end=start + 0.1))
    pm.instruments.append(inst)
    pm.write(str(path))


@pytest.mark.data_ops
def test_auto_tag_train(tmp_path: Path) -> None:
    pytest.importorskip("sklearn", reason="scikit-learn not installed")
    _mk_loop(tmp_path / "a.mid")
    runner = CliRunner()
    out = tmp_path / "m.pkl"
    res = runner.invoke(
        cli.cli,
        [
            "groove",
            "train",
            str(tmp_path),
            "--order",
            "1",
            "--out",
            str(out),
            "--auto-tag",
            "--no-progress",
        ],
    )
    assert res.exit_code == 0
    assert out.exists()
