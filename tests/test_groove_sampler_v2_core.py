from pathlib import Path
from unittest import mock

import numpy as np
import pretty_midi

from utilities import groove_sampler_v2, memmap_utils


def _make_loop(path: Path) -> None:
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=0, is_drum=True)
    for i in range(4):
        start = i * 0.25
        inst.notes.append(pretty_midi.Note(velocity=100, pitch=36, start=start, end=start + 0.05))
        inst.notes.append(pretty_midi.Note(velocity=100, pitch=38, start=start, end=start + 0.05))
        inst.notes.append(pretty_midi.Note(velocity=100, pitch=42, start=start, end=start + 0.05))
    pm.instruments.append(inst)
    pm.write(str(path))


def test_parallel_invocation(tmp_path: Path) -> None:
    _make_loop(tmp_path / "a.mid")
    with mock.patch("utilities.groove_sampler_v2.Parallel") as par, mock.patch(
        "utilities.groove_sampler_v2.delayed", side_effect=lambda f: f
    ):
        par.return_value.__call__ = lambda funcs: [f() for f in funcs]
        groove_sampler_v2.train(tmp_path, n_jobs=2)
        par.assert_called_with(n_jobs=2)


def test_memmap_creation(tmp_path: Path) -> None:
    _make_loop(tmp_path / "b.mid")
    model = groove_sampler_v2.train(tmp_path, memmap_dir=tmp_path)
    path = tmp_path / "prob_order0.mmap"
    assert path.exists()
    mm = memmap_utils.load_memmap(path, shape=(len(model.ctx_maps[0]), len(model.idx_to_state)))
    assert mm.dtype == np.float32
    assert mm.shape == (len(model.ctx_maps[0]), len(model.idx_to_state))


def test_collision_no_kick_snare_same_tick(tmp_path: Path) -> None:
    _make_loop(tmp_path / "c.mid")
    model = groove_sampler_v2.train(tmp_path)
    events = groove_sampler_v2.generate_events(model, bars=1, seed=0)
    for off in {ev["offset"] for ev in events}:
        insts = [e["instrument"] for e in events if e["offset"] == off]
        assert not ("kick" in insts and "snare" in insts)


def test_velocity_condition_soft(tmp_path: Path) -> None:
    _make_loop(tmp_path / "d.mid")
    model = groove_sampler_v2.train(tmp_path)
    events = groove_sampler_v2.generate_events(model, bars=1, cond_velocity="soft")
    assert all(ev["velocity_factor"] <= 0.8 for ev in events)

