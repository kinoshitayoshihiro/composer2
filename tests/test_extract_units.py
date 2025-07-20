from pathlib import Path

import pretty_midi
import pytest

from utilities.articulation_csv import extract_from_midi
from utilities.duration_bucket import to_bucket


def test_extract_units(tmp_path: Path) -> None:
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(0)
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=0.5))
    pm.instruments.append(inst)
    midi = tmp_path / "note.mid"
    pm.write(str(midi))
    df = extract_from_midi(midi)
    row = df.loc[0]
    assert row["track_id"] == 0
    assert row["pitch"] == 60
    assert row["onset"] == 0.0
    assert row["duration"] == 1.0
    assert row["velocity"] == pytest.approx(100 / 127.0)
    assert row["pedal_state"] == 0
    assert to_bucket(row["duration"]) == 2
