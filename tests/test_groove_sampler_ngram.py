import random
from pathlib import Path

import pretty_midi
from utilities import groove_sampler


def _create_loop_midi(tmp_path: Path) -> Path:
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=0, is_drum=True)
    for i in range(4):
        inst.notes.append(
            pretty_midi.Note(velocity=100, pitch=36, start=i * 0.5, end=i * 0.5 + 0.1)
        )
    pm.instruments.append(inst)
    midi_path = tmp_path / "loop.mid"
    pm.write(str(midi_path))
    return midi_path


def test_given_unseen_context_when_sample_next_then_fallback_to_unigram(tmp_path: Path):
    _create_loop_midi(tmp_path)
    model = groove_sampler.load_grooves(tmp_path, n=3)
    state = groove_sampler.sample_next(
        [(0, "kick"), (8, "snare")], model, random.Random(0)
    )
    assert state in model["prob"][0][()]


def test_given_n_value_when_load_grooves_then_stored(tmp_path: Path):
    _create_loop_midi(tmp_path)
    model = groove_sampler.load_grooves(tmp_path, n=4)
    assert model["n"] == 4


def test_given_zero_jitter_when_generate_bar_then_velocity_constant(tmp_path: Path):
    _create_loop_midi(tmp_path)
    model = groove_sampler.load_grooves(tmp_path)
    events = groove_sampler.generate_bar(
        [],
        model,
        random.Random(0),
        resolution=16,
        velocity_jitter=lambda r: 0,
    )
    assert events
    assert all(ev["velocity_factor"] == 1.0 for ev in events)
