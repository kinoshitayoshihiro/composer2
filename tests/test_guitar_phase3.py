import shutil
import xml.etree.ElementTree as ET

import pytest
from music21 import chord, harmony

from generator.guitar_generator import (
    EXEC_STYLE_ARPEGGIO_PATTERN,
    EXEC_STYLE_BLOCK_CHORD,
    EXEC_STYLE_POWER_CHORDS,
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
    gen.compose(section_data={
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
    content = path.read_text().splitlines()
    assert len(content) == 7
    assert content[1].startswith("e|")
    assert content[-1].startswith("E|")
    assert content[0].startswith("# Q=")


def test_export_musicxml_tab(_basic_gen, tmp_path):
    gen = _basic_gen()
    gen.compose(section_data={
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


def test_stroke_velocity_factor_override(_basic_gen):
    cs = harmony.ChordSymbol("C")
    gen = _basic_gen(stroke_velocity_factor={"DOWN": 1.2, "UP": 0.8})
    gen.default_velocity_curve = None
    base = gen._create_notes_from_event(cs, {"execution_style": EXEC_STYLE_BLOCK_CHORD}, {}, 1.0, 80)
    down = gen._create_notes_from_event(cs, {"execution_style": EXEC_STYLE_BLOCK_CHORD}, {"current_event_stroke": "down"}, 1.0, 80)
    up = gen._create_notes_from_event(cs, {"execution_style": EXEC_STYLE_BLOCK_CHORD}, {"current_event_stroke": "up"}, 1.0, 80)
    base_v = base[0].notes[0].volume.velocity if isinstance(base[0], chord.Chord) else base[0].volume.velocity
    vel_down = down[0].notes[0].volume.velocity if isinstance(down[0], chord.Chord) else down[0].volume.velocity
    vel_up = up[0].notes[0].volume.velocity if isinstance(up[0], chord.Chord) else up[0].volume.velocity
    assert vel_down > base_v > vel_up


def test_export_musicxml_tab_lily(_basic_gen, tmp_path, monkeypatch):
    gen = _basic_gen()
    monkeypatch.setattr(shutil, "which", lambda x: None)
    gen.compose(
        section_data={
            "section_name": "A",
            "q_length": 1.0,
            "humanized_duration_beats": 1.0,
            "original_chord_label": "C",
            "chord_symbol_for_voicing": "C",
            "part_params": {},
            "musical_intent": {},
            "shared_tracks": {},
        }
    )
    path = tmp_path / "tab.ly"
    gen.export_musicxml_tab(str(path), format="lily")
    text = path.read_text()
    assert "TabStaff" in text
    assert text.splitlines()[0].strip() == "\\new TabStaff"


def test_velocity_curve_interpolation(_basic_gen):
    curve7 = [0, 20, 40, 60, 80, 100, 120]
    gen = _basic_gen(default_velocity_curve=curve7)
    assert len(gen.default_velocity_curve) == 128


def test_velocity_curve_interpolation_fallback(monkeypatch):
    import importlib
    import sys

    monkeypatch.setitem(sys.modules, "numpy", None)
    monkeypatch.setitem(sys.modules, "scipy", None)
    vc = importlib.reload(importlib.import_module("utilities.velocity_curve"))

    curve7 = [0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0]

    def linear_expected():
        pts = [i / 6 for i in range(7)]
        out = []
        for i in range(128):
            x = i / 127
            for j in range(6):
                if pts[j] <= x <= pts[j + 1]:
                    y = curve7[j] + (curve7[j + 1] - curve7[j]) * (x - pts[j]) / (
                        pts[j + 1] - pts[j]
                    )
                    out.append(y)
                    break
        return out

    result = vc.interpolate_7pt(curve7)
    assert len(result) == 128
    assert result == pytest.approx(linear_expected())
    importlib.reload(vc)


def test_velocity_curve_len3(_basic_gen):
    curve3 = [0, 64, 127]
    gen = _basic_gen(default_velocity_curve=curve3)
    assert len(gen.default_velocity_curve) == 128
    assert gen.default_velocity_curve[0] == 0
    assert gen.default_velocity_curve[-1] == 127


def test_velocity_curve_dict_spec(_basic_gen):
    cfg = {"preset": "crescendo"}
    gen = _basic_gen(default_velocity_curve=cfg)
    assert isinstance(gen.default_velocity_curve, list)


def test_stroke_factor_block(_basic_gen):
    cs = harmony.ChordSymbol("C")
    gen = _basic_gen()
    gen.default_velocity_curve = None
    base = gen._create_notes_from_event(cs, {"execution_style": EXEC_STYLE_BLOCK_CHORD}, {}, 1.0, 80)
    down = gen._create_notes_from_event(cs, {"execution_style": EXEC_STYLE_BLOCK_CHORD}, {"current_event_stroke": "down"}, 1.0, 80)
    up = gen._create_notes_from_event(cs, {"execution_style": EXEC_STYLE_BLOCK_CHORD}, {"current_event_stroke": "up"}, 1.0, 80)
    base_v = base[0].notes[0].volume.velocity if isinstance(base[0], chord.Chord) else base[0].volume.velocity
    vel_down = down[0].notes[0].volume.velocity if isinstance(down[0], chord.Chord) else down[0].volume.velocity
    vel_up = up[0].notes[0].volume.velocity if isinstance(up[0], chord.Chord) else up[0].volume.velocity
    assert vel_down > base_v > vel_up


def test_svf_power_chord(_basic_gen):
    cs = harmony.ChordSymbol("C")
    gen = _basic_gen()
    gen.default_velocity_curve = None
    base = gen._create_notes_from_event(cs, {"execution_style": EXEC_STYLE_POWER_CHORDS}, {}, 1.0, 80)
    down = gen._create_notes_from_event(cs, {"execution_style": EXEC_STYLE_POWER_CHORDS}, {"current_event_stroke": "down"}, 1.0, 80)
    base_v = base[0].notes[0].volume.velocity if isinstance(base[0], chord.Chord) else base[0].volume.velocity
    vel_down = down[0].notes[0].volume.velocity if isinstance(down[0], chord.Chord) else down[0].volume.velocity
    assert vel_down >= int(round(base_v * 1.2))

