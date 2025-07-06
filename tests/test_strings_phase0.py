from music21 import instrument, harmony, pitch, converter

import importlib.util
import sys
import types
from pathlib import Path

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
        "q_length": 4.0,
        "humanized_duration_beats": 4.0,
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


def test_basic_chord_returns_five_parts():
    gen = _gen()
    parts = gen.compose(section_data=_basic_section())
    assert set(parts.keys()) == {
        "contrabass",
        "violoncello",
        "viola",
        "violin_ii",
        "violin_i",
    }
    for p in parts.values():
        notes = list(p.flatten().notes)
        assert len(notes) == 1


def test_note_ranges_within_limits():
    gen = _gen()
    parts = gen.compose(section_data=_basic_section())
    ranges = {
        "violin_i": ("G3", "D7"),
        "violin_ii": ("G3", "D7"),
        "viola": ("C3", "A5"),
        "violoncello": ("C2", "E4"),
        "contrabass": ("C1", "C3"),
    }
    for name, (lo, hi) in ranges.items():
        n = list(parts[name].flatten().notes)[0]
        assert pitch.Pitch(lo).midi <= n.pitch.midi <= pitch.Pitch(hi).midi


def test_manual_voice_allocation():
    gen = _gen(voice_allocation={"violin_i": 0})
    parts = gen.compose(section_data=_basic_section())
    n = list(parts["violin_i"].flatten().notes)[0]
    assert n.pitch.name == harmony.ChordSymbol("C").root().name


def test_voicing_mode_open_and_spread():
    diff = {}
    for mode in ["close", "open", "spread"]:
        gen = _gen(voicing_mode=mode)
        parts = gen.compose(section_data=_basic_section())
        diff[mode] = (
            parts["violin_i"].flatten().notes[0].pitch.midi
            - parts["contrabass"].flatten().notes[0].pitch.midi
        )
    assert diff["spread"] > diff["close"]


def test_long_duration_ties_and_export(tmp_path):
    section = _basic_section()
    section["q_length"] = 8.0
    gen = _gen()
    gen.compose(section_data=section)
    out = tmp_path / "out.xml"
    gen.export_musicxml(str(out))
    sc = converter.parse(str(out))
    cb_notes = list(sc.parts[0].recurse().notes)
    assert cb_notes[0].tie.type == "start"
    assert cb_notes[-1].tie is not None


def test_missing_voice_returns_rest():
    gen = _gen(voice_allocation={"violin_i": -1})
    parts = gen.compose(section_data=_basic_section())
    assert parts["violin_i"].flatten().notesAndRests[0].isRest
