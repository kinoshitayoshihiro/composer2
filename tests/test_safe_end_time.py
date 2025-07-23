import importlib.util
import sys
from pathlib import Path

import numpy as np
import pretty_midi

spec = importlib.util.spec_from_file_location(
    "utilities.midi_utils",
    str(Path(__file__).resolve().parents[1] / "utilities" / "midi_utils.py"),
)
assert spec and spec.loader
midi_utils = importlib.util.module_from_spec(spec)
sys.modules["utilities.midi_utils"] = midi_utils
spec.loader.exec_module(midi_utils)
safe_end_time = midi_utils.safe_end_time


def _build_pm() -> pretty_midi.PrettyMIDI:
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=0)
    inst.notes.append(pretty_midi.Note(velocity=90, pitch=60, start=0.0, end=0.5))
    pm.instruments.append(inst)
    pm.lyrics.append(pretty_midi.Lyric(text="a", time=0.8))
    return pm


def test_safe_end_time_list(monkeypatch):
    pm = _build_pm()

    def fake_get_tempo_changes():
        return [0.0, 1.2], [120.0, 120.0]

    monkeypatch.setattr(pm, "get_tempo_changes", fake_get_tempo_changes)
    assert safe_end_time(pm) == 1.2


def test_safe_end_time_ndarray(monkeypatch):
    pm = _build_pm()

    def fake_get_tempo_changes():
        return np.array([0.0, 1.5]), np.array([120.0, 120.0])

    monkeypatch.setattr(pm, "get_tempo_changes", fake_get_tempo_changes)
    assert safe_end_time(pm) == 1.5
