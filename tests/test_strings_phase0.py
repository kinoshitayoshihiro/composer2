from music21 import instrument, harmony, pitch, converter, chord, stream, note, tie, meter

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


def test_divisi_third():
    gen = _gen(divisi={"violin_i": "third"})
    parts = gen.compose(section_data=_basic_section())
    chord_obj = parts["violin_i"].flatten().notes[0]
    assert isinstance(chord_obj, chord.Chord)
    interval_semitones = abs(chord_obj.pitches[1].midi - chord_obj.pitches[0].midi)
    assert interval_semitones in (3, 4)


def test_long_ties_total_duration():
    section = _basic_section()
    section["q_length"] = 8.0
    gen = _gen()
    parts = gen.compose(section_data=section)
    notes = list(parts["contrabass"].flatten().notes)
    total = sum(n.quarterLength for n in notes)
    assert abs(total - section["q_length"]) < 1e-6


def test_missing_voice_returns_rest():
    gen = _gen(voice_allocation={"violin_i": -1})
    parts = gen.compose(section_data=_basic_section())
    assert parts["violin_i"].flatten().notesAndRests[0].isRest


def test_long_ties_big():
    section = _basic_section()
    section["q_length"] = 64.0
    gen = _gen()
    parts = gen.compose(section_data=section)
    notes = list(parts["contrabass"].flatten().notes)
    assert abs(sum(n.quarterLength for n in notes) - section["q_length"]) < 1e-6


def test_velocity_clamping():
    gen = _gen()
    info_low = strings_module._SectionInfo("tmp", instrument.Violin(), "C4", "C6", 0.2)
    info_high = strings_module._SectionInfo("tmp", instrument.Violin(), "C4", "C6", 0.9)
    v_low = gen._velocity_for(info_low)
    v_high = gen._velocity_for(info_high)
    assert 1 <= v_low <= 127
    assert 1 <= v_high <= 127
    assert v_low < v_high


def test_merge_identical_bars_duration():
    gen = _gen()
    p = stream.Part()
    p.insert(0, instrument.Violin())
    p.insert(0, meter.TimeSignature("4/4"))
    n1 = note.Note("C4", quarterLength=4.0)
    n1.tie = tie.Tie("start")
    n2 = note.Note("C4", quarterLength=4.0)
    n2.tie = tie.Tie("start")
    p.append(n1)
    p.append(n2)
    merged = gen._merge_identical_bars(p)
    assert abs(merged.highestTime - p.highestTime) < 1e-6
    assert len(list(merged.flatten().notes)) == 1


def test_avoid_low_open_strings():
    gen = _gen(avoid_low_open_strings=True, voice_allocation={"viola": 0})
    parts = gen.compose(section_data=_basic_section())
    viola_note = parts["viola"].flatten().notes[0]
    assert pitch.Pitch("C4").midi <= viola_note.pitch.midi


def test_expression_cc_added():
    section = _basic_section()
    section["q_length"] = 4.0
    gen = _gen()
    parts = gen.compose(section_data=section)
    cc = parts["contrabass"].extra_cc
    assert cc and cc[-1]["val"] == 80
