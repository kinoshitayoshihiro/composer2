import pytest
import music21
from music21 import harmony
import xml.etree.ElementTree as ET
from generator.guitar_generator import (
    EXEC_STYLE_ARPEGGIO_PATTERN,
)





def test_arpeggio_pattern_offsets(_basic_gen):
    gen = _basic_gen()
    cs = harmony.ChordSymbol("C")
    pattern = {"execution_style": EXEC_STYLE_ARPEGGIO_PATTERN, "string_order": [5,4,3,2]}
    notes = gen._create_notes_from_event(cs, pattern, {}, 2.0, 80)
    offs = [round(float(n.offset), 2) for n in notes]
    assert len(offs) == 4
    assert offs == [0.0, 0.5, 1.0, 1.5]


def test_position_lock_effect(_basic_gen):
    cs = harmony.ChordSymbol("C")
    pattern = {"execution_style": EXEC_STYLE_ARPEGGIO_PATTERN, "string_order": [5,4,3,2,1,0]}
    gen_free = _basic_gen()
    gen_lock = _basic_gen(position_lock=True, preferred_position=3)
    free_notes = gen_free._create_notes_from_event(cs, pattern, {}, 3.0, 80)
    lock_notes = gen_lock._create_notes_from_event(cs, pattern, {}, 3.0, 80)
    free_frets = [getattr(n, "fret", 0) for n in free_notes]
    lock_frets = [getattr(n, "fret", 0) for n in lock_notes]
    assert max(lock_frets) - min(lock_frets) <= 4
    assert min(lock_frets) >= 1 and max(lock_frets) <= 5
    assert (min(free_frets) < 1) or (max(free_frets) > 5)


def test_export_tab_enhanced(_basic_gen, tmp_path):
    gen = _basic_gen()
    part = gen.compose(section_data={
        "section_name": "A",
        "q_length": 1.0,
        "humanized_duration_beats": 1.0,
        "original_chord_label": "C",
        "chord_symbol_for_voicing": "C",
        "part_params": {},
        "musical_intent": {},
        "shared_tracks": {},
    })
    path = tmp_path / "tab.txt"
    gen.export_tab_enhanced(str(path))
    content = path.read_text()
    assert "|" in content


def test_export_musicxml_tab(_basic_gen, tmp_path):
    gen = _basic_gen()
    part = gen.compose(section_data={
        "section_name": "A",
        "q_length": 1.0,
        "humanized_duration_beats": 1.0,
        "original_chord_label": "C",
        "chord_symbol_for_voicing": "C",
        "part_params": {},
        "musical_intent": {},
        "shared_tracks": {},
    })
    path = tmp_path / "tab.xml"
    gen.export_musicxml_tab(str(path))
    tree = ET.parse(path)
    root = tree.getroot()
    strings = root.findall(".//string")
    frets = root.findall(".//fret")
    assert len(strings) == len(frets) > 0


def test_hybrid_pattern_types(_basic_gen):
    gen = _basic_gen()
    gen.part_parameters["hybrid"] = {
        "pattern": [
            {
                "offset": 0.0,
                "duration": 1.0,
                "pattern_type": "strum",
                "execution_style": "strum_basic",
            },
            {
                "offset": 1.0,
                "duration": 1.0,
                "pattern_type": "arpeggio",
                "string_order": [5, 4, 3, 2],
                "execution_style": EXEC_STYLE_ARPEGGIO_PATTERN,
            },
        ],
        "reference_duration_ql": 1.0,
    }

    section = {
        "section_name": "A",
        "q_length": 2.0,
        "humanized_duration_beats": 2.0,
        "original_chord_label": "C",
        "chord_symbol_for_voicing": "C",
        "part_params": {"g": {"guitar_rhythm_key": "hybrid"}},
        "musical_intent": {},
        "shared_tracks": {},
    }
    part = gen.compose(section_data=section)
    offsets = [round(float(n.offset), 2) for n in part.flatten().notes]
    assert offsets == sorted(offsets)


def test_arpeggio_note_overlap(_basic_gen):
    gen = _basic_gen()
    cs = harmony.ChordSymbol("C")
    pattern = {
        "execution_style": EXEC_STYLE_ARPEGGIO_PATTERN,
        "string_order": [5, 4, 3, 2],
        "arpeggio_note_spacing_ms": 250,
    }
    notes = gen._create_notes_from_event(cs, pattern, {}, 2.0, 80)
    for a, b in zip(notes, notes[1:]):
        assert a.offset + a.quarterLength <= b.offset + 1e-3


def test_string_order_loop(_basic_gen):
    gen = _basic_gen()
    cs = harmony.ChordSymbol("C7")
    pattern = {
        "execution_style": EXEC_STYLE_ARPEGGIO_PATTERN,
        "string_order": [5],
        "arpeggio_note_spacing_ms": 250,
    }
    notes = gen._create_notes_from_event(cs, pattern, {}, 2.0, 80)
    assert len(notes) == 4


def test_string_order_missing(_basic_gen):
    gen = _basic_gen(strict_string_order=True)
    cs = harmony.ChordSymbol("C")
    pattern = {"execution_style": EXEC_STYLE_ARPEGGIO_PATTERN}
    notes = gen._create_notes_from_event(cs, pattern, {}, 2.0, 80)
    assert len(notes) > 0

