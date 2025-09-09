import importlib.util
from pathlib import Path

from music21 import instrument, note

from generator.piano_template_generator import PianoTemplateGenerator

spec = importlib.util.spec_from_file_location(
    "voicing_density",
    Path(__file__).resolve().parents[1] / "generator" / "voicing_density.py",
)
vd_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vd_module)
VoicingDensityEngine = vd_module.VoicingDensityEngine


def make_gen():
    return PianoTemplateGenerator(
        part_name="piano",
        default_instrument=instrument.Piano(),
        global_tempo=120,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
    )


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


def test_density_scale_integration():
    gen = make_gen()
    base_section = {
        "q_length": 4.0,
        "chord_symbol_for_voicing": "C",
        "groove_kicks": [],
        "musical_intent": {"intensity": "medium"},
    }

    orig = gen.compose(section_data=base_section)["piano_rh"]
    orig_len = len(list(orig.flatten().notes))

    low_sec = base_section.copy()
    low_sec["musical_intent"] = {"intensity": "low"}
    low_part = gen.compose(section_data=low_sec)["piano_rh"]
    low_len = len(list(low_part.flatten().notes))
    assert low_len < orig_len * 0.6

    high_sec = base_section.copy()
    high_sec["musical_intent"] = {"intensity": "high"}
    high_part = gen.compose(section_data=high_sec)["piano_rh"]
    high_len = len(list(high_part.flatten().notes))
    assert high_len > orig_len * 1.1
