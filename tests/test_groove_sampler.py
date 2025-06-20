import random
from pathlib import Path

import pretty_midi
from utilities import groove_sampler


def _create_loop_midi(tmp_path: Path) -> Path:
    """Create a simple 4-beat kick loop and return its path."""
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


def test_given_loop_when_load_grooves_then_model_contains_transitions(tmp_path: Path):
    _create_loop_midi(tmp_path)
    model = groove_sampler.load_grooves(tmp_path)
    assert model["n"] == 2
    assert model["resolution"] == 16
    key = ((0, "kick"),)
    assert key in model["prob"][1]
    assert (16, "kick") in model["prob"][1][key]


def test_given_n_parameter_when_load_grooves_then_stored(tmp_path: Path):
    _create_loop_midi(tmp_path)
    model = groove_sampler.load_grooves(tmp_path, n=3)
    assert model["n"] == 3


def test_given_context_when_sample_next_then_returns_expected(tmp_path: Path):
    # create alternating kick-snare pattern
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=0, is_drum=True)
    pitches = [36, 38, 36, 38]
    for i, p in enumerate(pitches):
        inst.notes.append(
            pretty_midi.Note(velocity=100, pitch=p, start=i * 0.5, end=i * 0.5 + 0.1)
        )
    pm.instruments.append(inst)
    midi_path = tmp_path / "alt.mid"
    pm.write(str(midi_path))
    model = groove_sampler.load_grooves(tmp_path)
    next_state = groove_sampler.sample_next([(0, "kick")], model, random.Random(0))
    assert next_state == (16, "snare")


def test_given_model_when_generate_bar_then_events_valid(tmp_path: Path):
    _create_loop_midi(tmp_path)
    model = groove_sampler.load_grooves(tmp_path, resolution=32)
    events = groove_sampler.generate_bar([], model, random.Random(0), resolution=32)
    assert len(events) > 0
    for e in events:
        valid = {lbl for d in model["prob"].values() for ctx in d for _, lbl in ctx}
        assert e["instrument"] in valid
        assert abs(e["offset"] * 32 - round(e["offset"] * 32)) < 1e-6


def test_given_default_jitter_when_generate_bar_then_in_range(tmp_path: Path):
    rng = random.Random(0)
    _create_loop_midi(tmp_path)
    model = groove_sampler.load_grooves(tmp_path)
    events = groove_sampler.generate_bar([], model, rng, resolution=16)
    vals = {e["velocity_factor"] for e in events}
    assert all(0.95 <= v <= 1.05 for v in vals)
