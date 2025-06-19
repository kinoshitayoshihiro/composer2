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


def test_model_build(tmp_path: Path):
    _create_loop_midi(tmp_path)
    model = groove_sampler.load_grooves(tmp_path)
    assert model["n"] == 2
    key = (0, "kick")
    assert key in model["freq"]
    assert model["freq"][key][(4, "kick")] == 1


def test_ngram_param(tmp_path: Path):
    _create_loop_midi(tmp_path)
    model = groove_sampler.load_grooves(tmp_path, n=3)
    assert model["n"] == 3


def test_sample_next_deterministic(tmp_path: Path):
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
    next_state = groove_sampler.sample_next((0, "kick"), model, random.Random(0))
    assert next_state == (4, "snare")


def test_generate_bar(tmp_path: Path):
    _create_loop_midi(tmp_path)
    model = groove_sampler.load_grooves(tmp_path)
    events = groove_sampler.generate_bar([], model, random.Random(0), resolution=16)
    assert len(events) > 0
    assert max(e["offset"] for e in events) < 1.0
    for e in events:
        assert e["instrument"] in {lbl for _, lbl in model["freq"].keys()}


def test_velocity_jitter_range(tmp_path: Path):
    rng = random.Random(0)
    _create_loop_midi(tmp_path)
    model = groove_sampler.load_grooves(tmp_path)
    events = groove_sampler.generate_bar([], model, rng, resolution=16)
    vals = {e["velocity_factor"] for e in events}
    assert all(0.95 <= v <= 1.05 for v in vals)
