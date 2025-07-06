from music21 import note
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "voicing_density",
    Path(__file__).resolve().parents[1] / "generator" / "voicing_density.py",
)
vd_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vd_module)
VoicingDensityEngine = vd_module.VoicingDensityEngine


def make_notes(n):
    part = []
    for i in range(n):
        part.append(note.Note('C4', quarterLength=1.0, offset=float(i)))
    return part


def test_low_density_reduction():
    eng = VoicingDensityEngine()
    notes = make_notes(8)
    out = eng.scale_density(notes, 'low')
    assert 3 <= len(out) <= 5


def test_high_density_expansion():
    eng = VoicingDensityEngine()
    notes = make_notes(10)
    out = eng.scale_density(notes, 'high')
    assert len(out) >= 11
