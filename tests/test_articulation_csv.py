from pathlib import Path

import pretty_midi

from utilities.articulation_csv import ArticRow, extract_from_midi


def create_midi() -> pretty_midi.PrettyMIDI:
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=0)
    inst.notes.append(pretty_midi.Note(start=0.0, end=0.1, pitch=60, velocity=100))
    inst.notes.append(pretty_midi.Note(start=0.3, end=0.4, pitch=60, velocity=100))
    inst.notes.append(pretty_midi.Note(start=0.8, end=0.9, pitch=60, velocity=100))
    inst.notes.append(pretty_midi.Note(start=1.2, end=1.3, pitch=60, velocity=100))
    inst.control_changes.append(pretty_midi.ControlChange(number=64, value=50, time=0.25))
    inst.control_changes.append(pretty_midi.ControlChange(number=64, value=100, time=0.75))
    inst.control_changes.append(pretty_midi.ControlChange(number=64, value=0, time=1.0))
    pm.instruments.append(inst)
    return pm


def test_extract_half_and_before(tmp_path: Path) -> None:
    pm = create_midi()
    midi_path = tmp_path / "test.mid"
    pm.write(str(midi_path))
    rows = extract_from_midi(midi_path)
    assert [r.pedal_state for r in rows] == [0, 2, 1, 0]
    assert all(isinstance(r, ArticRow) for r in rows)
