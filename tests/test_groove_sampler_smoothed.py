import random
from pathlib import Path

import pretty_midi
from utilities import groove_sampler


def _make_loop(path: Path) -> None:
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=0, is_drum=True)
    pattern = [36, 38, 36, 38, 36]
    for i, p in enumerate(pattern):
        inst.notes.append(pretty_midi.Note(velocity=100, pitch=p, start=i * 0.5, end=i * 0.5 + 0.1))
    pm.instruments.append(inst)
    pm.write(str(path / "loop.mid"))


def test_smoothed_backoff(tmp_path: Path):
    _make_loop(tmp_path)
    model = groove_sampler.load_grooves(tmp_path, n=5, smoothing=0.5, resolution=16)
    ctx = [(0, "kick"), (8, "snare"), (16, "kick"), (24, "snare")]
    assert tuple(ctx) not in model["prob"][4]
    state = groove_sampler.sample_next(ctx, model, random.Random(0))
    assert state is not None
