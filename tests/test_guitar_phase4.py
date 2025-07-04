import pytest
from music21 import harmony, articulations
from generator.guitar_generator import GuitarGenerator
from tests.conftest import _basic_gen


def test_phrase_marks_crescendo(_basic_gen):
    gen = _basic_gen()
    gen.part_parameters["qpat"] = {
        "pattern": [
            {"offset": 0.0, "duration": 1.0},
            {"offset": 1.0, "duration": 1.0},
            {"offset": 2.0, "duration": 1.0},
            {"offset": 3.0, "duration": 1.0},
        ],
        "reference_duration_ql": 4.0,
    }
    sec = {
        "section_name": "A",
        "q_length": 4.0,
        "humanized_duration_beats": 4.0,
        "original_chord_label": "C",
        "chord_symbol_for_voicing": "C",
        "part_params": {"g": {"guitar_rhythm_key": "qpat"}},
        "phrase_marks": ["crescendo"],
        "musical_intent": {},
        "shared_tracks": {},
    }
    part = gen.compose(section_data=sec)
    notes = list(part.flatten().notes)
    half = len(notes) // 2
    avg1 = sum(n.volume.velocity for n in notes[:half]) / max(1, half)
    avg2 = sum(n.volume.velocity for n in notes[half:]) / max(1, len(notes[half:]))
    assert avg2 > avg1


def test_execution_style_harmonic(_basic_gen):
    gen = _basic_gen()
    notes = gen._create_notes_from_event(harmony.ChordSymbol("C"), {"execution_style": "harmonic"}, {}, 1.0, 80)
    elem = notes[0]
    if hasattr(elem, "notes"):
        elem = elem.notes[0]
    assert any(isinstance(a, articulations.Harmonic) for a in elem.articulations)


def test_pick_position_cc74(_basic_gen):
    gen = _basic_gen()
    gen.part_parameters["qpat"] = {
        "pattern": [{"offset": 0.0, "duration": 1.0}],
        "reference_duration_ql": 1.0,
    }
    sec1 = {
        "section_name": "A",
        "q_length": 1.0,
        "humanized_duration_beats": 1.0,
        "original_chord_label": "C",
        "chord_symbol_for_voicing": "C",
        "part_params": {"g": {"guitar_rhythm_key": "qpat", "pick_position": 0.2}},
        "musical_intent": {},
        "shared_tracks": {},
    }
    sec2 = {
        "section_name": "A",
        "q_length": 1.0,
        "humanized_duration_beats": 1.0,
        "original_chord_label": "C",
        "chord_symbol_for_voicing": "C",
        "part_params": {"g": {"guitar_rhythm_key": "qpat", "pick_position": 0.8}},
        "musical_intent": {},
        "shared_tracks": {},
    }
    p1 = gen.compose(section_data=sec1)
    p2 = gen.compose(section_data=sec2)
    val1 = [c["val"] for c in getattr(p1, "extra_cc", []) if c.get("cc") == 74][0]
    val2 = [c["val"] for c in getattr(p2, "extra_cc", []) if c.get("cc") == 74][0]
    assert val2 > val1


def test_fx_pick_position_cc74(_basic_gen):
    gen = _basic_gen()
    gen.part_parameters["qpat"] = {
        "pattern": [{"offset": 0.0, "duration": 1.0}],
        "reference_duration_ql": 1.0,
    }
    base = {
        "section_name": "A",
        "q_length": 1.0,
        "humanized_duration_beats": 1.0,
        "original_chord_label": "C",
        "chord_symbol_for_voicing": "C",
        "part_params": {"g": {"guitar_rhythm_key": "qpat"}},
    }
    sec1 = base | {"fx_params": {"pick_position": 0.2}}
    sec2 = base | {"fx_params": {"pick_position": 0.8}}
    p1 = gen.compose(section_data=sec1)
    p2 = gen.compose(section_data=sec2)
    val1 = [c["val"] for c in p1.extra_cc if c.get("cc") == 74][0]
    val2 = [c["val"] for c in p2.extra_cc if c.get("cc") == 74][0]
    assert val2 > val1


def test_style_hint_soft(_basic_gen):
    gen = _basic_gen()
    gen.part_parameters["qpat"] = {
        "pattern": [{"offset": 0.0, "duration": 1.0}],
        "reference_duration_ql": 1.0,
    }
    base_sec = {
        "section_name": "A",
        "q_length": 1.0,
        "humanized_duration_beats": 1.0,
        "original_chord_label": "C",
        "chord_symbol_for_voicing": "C",
        "part_params": {"g": {"guitar_rhythm_key": "qpat"}},
        "musical_intent": {},
        "shared_tracks": {},
    }
    soft_sec = base_sec | {"style_hint": "soft"}
    p_base = gen.compose(section_data=base_sec)
    p_soft = gen.compose(section_data=soft_sec)
    base_vel = sum(n.volume.velocity for n in p_base.flatten().notes) / len(p_base.flatten().notes)
    soft_vel = sum(n.volume.velocity for n in p_soft.flatten().notes) / len(p_soft.flatten().notes)
    assert soft_vel < base_vel


def test_execution_style_vibrato(_basic_gen):
    gen = _basic_gen()
    notes = gen._create_notes_from_event(harmony.ChordSymbol("C"), {"execution_style": "vibrato"}, {}, 1.0, 80)
    elem = notes[0]
    if hasattr(elem, "notes"):
        elem = elem.notes[0]
    assert any(isinstance(a, articulations.Shake) for a in elem.articulations)


def test_random_walk_cc(_basic_gen):
    gen = _basic_gen()
    gen.part_parameters["qpat"] = {
        "pattern": [{"offset": 0.0, "duration": 1.0}],
        "reference_duration_ql": 1.0,
    }
    sec = {
        "section_name": "A",
        "q_length": 4.0,
        "humanized_duration_beats": 4.0,
        "original_chord_label": "C",
        "chord_symbol_for_voicing": "C",
        "part_params": {"g": {"guitar_rhythm_key": "qpat"}},
        "musical_intent": {},
        "shared_tracks": {},
        "random_walk_cc": True,
    }
    part = gen.compose(section_data=sec)
    events = [e for e in getattr(part, "extra_cc", []) if e.get("cc") == 1]
    assert len(events) > 1


def test_execution_style_pinch_harmonic(_basic_gen):
    gen = _basic_gen()
    notes = gen._create_notes_from_event(
        harmony.ChordSymbol("C"), {"execution_style": "pinch_harmonic"}, {}, 1.0, 80
    )
    elem = notes[0]
    if hasattr(elem, "notes"):
        elem = elem.notes[0]
    assert any(
        a.__class__.__name__ == "PinchHarmonic" for a in elem.articulations
    )


def test_style_db_external_load(_basic_gen, tmp_path):
    style_file = tmp_path / "s.yaml"
    style_file.write_text(
        "mycurve:\n  velocity: [10, 20, 30, 40]\n  cc: [50, 60, 70, 80]\n",
        encoding="utf-8",
    )
    gen = _basic_gen(style_db_path=str(style_file))
    gen.part_parameters["qpat"] = {
        "pattern": [{"offset": 0.0, "duration": 1.0}],
        "reference_duration_ql": 1.0,
    }
    sec = {
        "section_name": "A",
        "q_length": 1.0,
        "humanized_duration_beats": 1.0,
        "original_chord_label": "C",
        "chord_symbol_for_voicing": "C",
        "part_params": {"g": {"guitar_rhythm_key": "qpat"}},
        "musical_intent": {},
        "shared_tracks": {},
        "style_hint": "mycurve",
    }
    part = gen.compose(section_data=sec)
    first_vel = part.flatten().notes[0].volume.velocity
    assert first_vel == 10
    assert any(e.get("val") == 50 for e in part.extra_cc)


def test_envelope_map_multi_cc(_basic_gen):
    gen = _basic_gen()
    gen.part_parameters["qpat"] = {
        "pattern": [{"offset": 0.0, "duration": 1.0}],
        "reference_duration_ql": 1.0,
    }
    sec = {
        "section_name": "A",
        "q_length": 1.0,
        "humanized_duration_beats": 1.0,
        "original_chord_label": "C",
        "chord_symbol_for_voicing": "C",
        "part_params": {"g": {"guitar_rhythm_key": "qpat"}},
        "musical_intent": {},
        "shared_tracks": {},
        "envelope_map": {0.0: {"type": "crescendo", "duration_ql": 1.0, "cc": [11, 72, 74]}},
    }
    part = gen.compose(section_data=sec)
    ccs = {e["cc"] for e in getattr(part, "extra_cc", [])}
    assert {11, 72, 74}.issubset(ccs)
