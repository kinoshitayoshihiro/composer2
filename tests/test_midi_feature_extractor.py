from pathlib import Path
import sys
import pretty_midi

sys.path.append(str(Path(__file__).resolve().parents[1]))
from tools.midi_feature_extractor import extract_features


def test_extract_features(tmp_path: Path):
    pm = pretty_midi.PrettyMIDI(initial_tempo=100)
    inst = pretty_midi.Instrument(program=0)
    note = pretty_midi.Note(velocity=80, pitch=60, start=0.0, end=1.0)
    inst.notes.append(note)
    pm.instruments.append(inst)
    midi_path = tmp_path / "test.mid"
    pm.write(str(midi_path))
    feats = extract_features(midi_path)
    assert round(feats["tempo_bpm"], 2) == 100
    assert feats["total_notes"] == 1
    assert round(feats["mean_velocity"], 1) == 80.0
    assert round(feats["note_density_per_sec"], 2) == 1.0
