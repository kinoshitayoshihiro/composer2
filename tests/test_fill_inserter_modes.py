import random
import pytest
from music21 import stream
from generator.drum_generator import FillInserter, HoldTie


def test_template_list_choice():
    lib = {"f": {"template": ["T1", "T2"], "length_beats": 1.0}}
    rng = random.Random(0)
    fi = FillInserter(lib, rng=rng)
    fi.drum_map = {"tom1": ("", 48), "tom2": ("", 47)}
    part = stream.Part(id="drums")
    fi.insert(part, {"q_length": 4.0, "absolute_offset": 0.0}, "f")
    chosen = random.Random(0).choice(["T1", "T2"])
    note_midi = fi.drum_map[{"T1": "tom1", "T2": "tom2"}[chosen]][1]
    assert [n.pitch.midi for n in part.flatten().notes] == [note_midi]


def test_legato_mode_ties():
    lib = {
        "leg": {
            "pattern": [
                {"instrument": "snare", "offset": 0.0},
                {"instrument": "snare", "offset": 0.5},
            ],
            "mode": "legato",
        }
    }
    fi = FillInserter(lib)
    fi.drum_map = {"snare": ("", 38)}
    part = stream.Part(id="drums")
    fi.insert(part, {"q_length": 4.0, "absolute_offset": 0.0}, "leg")
    notes = list(part.flatten().notes)
    assert isinstance(notes[0].tie, HoldTie)
    assert notes[0].duration.quarterLength == pytest.approx(0.5)


def test_velocity_curve_applied():
    lib = {
        "vel": {
            "pattern": [
                {"instrument": "snare", "offset": 0.0, "velocity_factor": 1.0},
                {"instrument": "snare", "offset": 1.0, "velocity_factor": 1.0},
            ],
            "velocity_curve": [0.5, 1.5],
        }
    }
    fi = FillInserter(lib)
    fi.drum_map = {"snare": ("", 38)}
    part = stream.Part(id="drums")
    fi.insert(part, {"q_length": 4.0, "absolute_offset": 0.0}, "vel")
    vels = [n.volume.velocity for n in part.flatten().notes]
    assert vels == [40, 120]


def test_base_velocity_override():
    lib = {
        "b": {
            "pattern": [
                {"instrument": "snare", "offset": 0.0, "velocity_factor": 1.0},
            ],
            "base_velocity": 100,
        }
    }
    fi = FillInserter(lib, base_velocity=70)
    fi.drum_map = {"snare": ("", 38)}
    part = stream.Part(id="drums")
    fi.insert(part, {"q_length": 4.0, "absolute_offset": 0.0}, "b")
    assert [n.volume.velocity for n in part.flatten().notes] == [100]
