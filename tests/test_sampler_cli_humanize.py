import io
from pathlib import Path

import pretty_midi
from click.testing import CliRunner

from utilities import groove_sampler_ngram as gs


def _make_loop(path: Path) -> None:
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=0, is_drum=True)
    for i in range(4):
        start = i * 0.25
        vel = 90 + i * 5
        inst.notes.append(pretty_midi.Note(velocity=vel, pitch=36, start=start, end=start + 0.05))
    pm.instruments.append(inst)
    pm.write(str(path))


def test_cli_humanize(tmp_path: Path) -> None:
    _make_loop(tmp_path / "a.mid")
    model = gs.train(tmp_path, order=1)
    gs.save(model, tmp_path / "m.pkl")
    runner = CliRunner()
    res = runner.invoke(
        gs.cli,
        ["sample", str(tmp_path / "m.pkl"), "-l", "1", "--seed", "0", "--humanize", "vel,micro"],
    )
    assert res.exit_code == 0
    pm = pretty_midi.PrettyMIDI(io.BytesIO(res.stdout_bytes))
    vels = [n.velocity for n in pm.instruments[0].notes]
    assert max(vels) != min(vels)
    step_ticks = gs.PPQ // 4
    for n in pm.instruments[0].notes:
        start_beats = n.start / 0.5
        ticks = round(start_beats * gs.PPQ)
        step = round(start_beats * 4)
        micro = ticks - step * step_ticks
        assert -30 <= micro <= 30


def test_cli_micro_bounds(tmp_path: Path) -> None:
    _make_loop(tmp_path / "a.mid")
    model = gs.train(tmp_path, order=1)
    gs.save(model, tmp_path / "m.pkl")
    runner = CliRunner()
    micro_max = 10
    res = runner.invoke(
        gs.cli,
        [
            "sample",
            str(tmp_path / "m.pkl"),
            "-l",
            "1",
            "--seed",
            "0",
            "--humanize",
            "micro",
            "--micro-max",
            str(micro_max),
        ],
    )
    assert res.exit_code == 0
    pm = pretty_midi.PrettyMIDI(io.BytesIO(res.stdout_bytes))
    step_ticks = gs.PPQ // 4
    for n in pm.instruments[0].notes:
        start_beats = n.start / 0.5
        ticks = round(start_beats * gs.PPQ)
        step = round(start_beats * 4)
        micro = ticks - step * step_ticks
        assert -micro_max <= micro <= micro_max

