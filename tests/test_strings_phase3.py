import sys
import types
import importlib.util
from pathlib import Path
from music21 import instrument, spanner, pitch

ROOT = Path(__file__).resolve().parents[1]
pkg = types.ModuleType("generator")
pkg.__path__ = [str(ROOT / "generator")]
sys.modules.setdefault("generator", pkg)

_MOD_PATH = ROOT / "generator" / "strings_generator.py"
spec = importlib.util.spec_from_file_location("generator.strings_generator", _MOD_PATH)
strings_module = importlib.util.module_from_spec(spec)
sys.modules["generator.strings_generator"] = strings_module
spec.loader.exec_module(strings_module)
StringsGenerator = strings_module.StringsGenerator


def _basic_section():
    return {
        "section_name": "A",
        "q_length": 2.0,
        "humanized_duration_beats": 2.0,
        "original_chord_label": "C",
        "chord_symbol_for_voicing": "C",
        "part_params": {},
        "musical_intent": {},
        "shared_tracks": {},
    }


def _gen(**kwargs):
    return StringsGenerator(
        global_settings={},
        default_instrument=instrument.Violin(),
        part_name="strings",
        global_tempo=120,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
        **kwargs,
    )


def test_auto_slur_created():
    gen = _gen()
    sec = _basic_section()
    sec["events"] = [{"duration": 1.0}, {"duration": 1.0}]
    parts = gen.compose(section_data=sec)
    slurs = [s for s in parts["violin_i"].spanners if isinstance(s, spanner.Slur)]
    assert len(slurs) == 1


def test_dynamic_envelope_cc():
    gen = _gen()
    sec = _basic_section()
    sec["dim_start"] = 50
    sec["dim_end"] = 70
    parts = gen.compose(section_data=sec)
    cc = parts["violin_i"].extra_cc
    assert cc[0]["val"] == 50
    assert cc[-1]["val"] == 70
    assert cc[-1]["time"] == sec["q_length"]


def test_auto_divisi_note_count():
    gen = _gen()
    sec = _basic_section()
    sec["chord_symbol_for_voicing"] = "C13"
    sec["original_chord_label"] = "C13"
    parts = gen.compose(section_data=sec)
    n_violin_ii = parts["violin_ii"].flatten().notes[0]
    n_viola = parts["viola"].flatten().notes[0]

    def _count(obj):
        return len(obj.pitches) if hasattr(obj, "pitches") else 1

    assert _count(n_violin_ii) > 1 or _count(n_viola) > 1


def test_fit_pitch_reduced_motion():
    res = StringsGenerator._fit_pitch(pitch.Pitch("E4"), 60, 80, 65)
    assert abs(res.midi - 65) <= 4
