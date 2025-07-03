import statistics
import pytest
import music21
from music21 import instrument, harmony
from generator.guitar_generator import (
    GuitarGenerator,
    EXEC_STYLE_STRUM_BASIC,
    GUITAR_STRUM_DELAY_QL,
)


def _basic_gen(**kwargs):
    return GuitarGenerator(
        global_settings={},
        default_instrument=instrument.Guitar(),
        part_name="g",
        global_tempo=120,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
        **kwargs
    )


def test_timing_jitter_ms_variation():
    gen = _basic_gen(timing_jitter_ms=20)
    gen.rng.seed(0)
    notes = gen._create_notes_from_event(
        harmony.ChordSymbol("C"),
        {"execution_style": EXEC_STYLE_STRUM_BASIC},
        {},
        1.0,
        80,
    )
    diffs = [n.offset - i * GUITAR_STRUM_DELAY_QL for i, n in enumerate(notes)]
    assert statistics.pstdev(diffs) > 0


def test_swing_ratio_applied():
    gen = _basic_gen()
    gen.part_parameters["pair"] = {
        "pattern": [
            {"offset": 0.0, "duration": 0.5},
            {"offset": 0.5, "duration": 0.5},
        ],
        "reference_duration_ql": 1.0,
    }
    section = {
        "section_name": "A",
        "q_length": 1.0,
        "humanized_duration_beats": 1.0,
        "original_chord_label": "C",
        "chord_symbol_for_voicing": "C",
        "part_params": {"g": {"guitar_rhythm_key": "pair"}, "swing_ratio": 0.6},
        "musical_intent": {},
        "shared_tracks": {},
    }
    part = gen.compose(section_data=section)
    notes = list(part.flatten().notes)
    offs = [round(float(n.offset), 2) for n in notes]
    assert offs[1] == pytest.approx(0.6, abs=0.01)
    assert round(notes[0].quarterLength, 2) == pytest.approx(0.4, abs=0.01)


def test_accent_map_velocity():
    gen = _basic_gen(accent_map={0: 10, 2: -5})
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
        "musical_intent": {},
        "shared_tracks": {},
    }
    part = gen.compose(section_data=sec)
    vels = [n.volume.velocity for n in part.flatten().notes]
    base0 = gen.default_velocity_curve[int(round(127 * 0 / 4))]
    base2 = gen.default_velocity_curve[int(round(127 * 0.5))]
    assert vels[0] == base0 + 10
    assert vels[2] == base2 - 5


def test_round_robin_channels():
    gen = _basic_gen(rr_channel_cycle=[10, 11])
    gen.part_parameters["tri"] = {
        "pattern": [
            {"offset": 0.0, "duration": 0.5},
            {"offset": 0.5, "duration": 0.5},
            {"offset": 1.0, "duration": 0.5},
        ],
        "reference_duration_ql": 1.5,
    }
    sec = {
        "section_name": "A",
        "q_length": 1.5,
        "humanized_duration_beats": 1.5,
        "original_chord_label": "C",
        "chord_symbol_for_voicing": "C",
        "part_params": {"g": {"guitar_rhythm_key": "tri"}},
        "musical_intent": {},
        "shared_tracks": {},
    }
    part = gen.compose(section_data=sec)
    chans = [getattr(n, "channel", None) for n in part.flatten().notes]
    assert chans[:3] == [10, 11, 10]


def test_velocity_curve():
    curve = [i + 1 for i in range(128)]
    gen = _basic_gen(default_velocity_curve=curve, accent_map={}, swing_ratio=0.5)
    gen.part_parameters["vc1"] = {
        "pattern": [
            {"offset": 0.0, "duration": 1.0},
        ],
        "reference_duration_ql": 1.0,
    }
    sec = {
        "section_name": "A",
        "q_length": 1.0,
        "humanized_duration_beats": 1.0,
        "original_chord_label": "C",
        "chord_symbol_for_voicing": "C",
        "part_params": {"g": {"guitar_rhythm_key": "vc1"}},
        "musical_intent": {},
        "shared_tracks": {},
    }
    part = gen.compose(section_data=sec)
    vels = [n.volume.velocity for n in part.flatten().notes]
    assert vels[0] == curve[0]


def test_jitter_mode_gauss_vs_uniform():
    gen_u = _basic_gen(timing_jitter_ms=20, timing_jitter_mode="uniform")
    gen_g = _basic_gen(timing_jitter_ms=20, timing_jitter_mode="gauss")
    gen_u.rng.seed(1)
    gen_g.rng.seed(1)
    notes_u = gen_u._create_notes_from_event(
        harmony.ChordSymbol("C"),
        {"execution_style": EXEC_STYLE_STRUM_BASIC},
        {},
        1.0,
        80,
    )
    notes_g = gen_g._create_notes_from_event(
        harmony.ChordSymbol("C"),
        {"execution_style": EXEC_STYLE_STRUM_BASIC},
        {},
        1.0,
        80,
    )
    diffs_u = [n.offset - i * GUITAR_STRUM_DELAY_QL for i, n in enumerate(notes_u)]
    diffs_g = [n.offset - i * GUITAR_STRUM_DELAY_QL for i, n in enumerate(notes_g)]
    assert statistics.pstdev(diffs_g) > statistics.pstdev(diffs_u)


def test_swing_subdiv_shuffle():
    gen = _basic_gen()
    gen.part_parameters["tri"] = {
        "pattern": [
            {"offset": 0.0, "duration": 1/3},
            {"offset": 1/3, "duration": 1/3},
        ],
        "reference_duration_ql": 2/3,
    }
    sec = {
        "section_name": "A",
        "q_length": 2/3,
        "humanized_duration_beats": 2/3,
        "original_chord_label": "C",
        "chord_symbol_for_voicing": "C",
        "part_params": {"g": {"guitar_rhythm_key": "tri"}, "swing_ratio": 0.66, "swing_subdiv": 12},
        "musical_intent": {},
        "shared_tracks": {},
    }
    part = gen.compose(section_data=sec)
    offs = [round(float(n.offset), 3) for n in part.flatten().notes]
    assert offs[1] == pytest.approx(0.44, abs=0.01)


def test_swing_subdiv_zero():
    gen = _basic_gen()
    gen.part_parameters["pair"] = {
        "pattern": [
            {"offset": 0.0, "duration": 0.5},
            {"offset": 0.5, "duration": 0.5},
        ],
        "reference_duration_ql": 1.0,
    }
    sec = {
        "section_name": "A",
        "q_length": 1.0,
        "humanized_duration_beats": 1.0,
        "original_chord_label": "C",
        "chord_symbol_for_voicing": "C",
        "part_params": {"g": {"guitar_rhythm_key": "pair"}, "swing_ratio": 0.7, "swing_subdiv": 0},
        "musical_intent": {},
        "shared_tracks": {},
    }
    part = gen.compose(section_data=sec)
    offs = [round(float(n.offset), 2) for n in part.flatten().notes]
    assert offs == [0.0, 0.5]


def test_strum_delay_jitter_ms():
    gen = _basic_gen(strum_delay_jitter_ms=10)
    gen.rng.seed(0)
    notes = gen._create_notes_from_event(
        harmony.ChordSymbol("C"),
        {"execution_style": EXEC_STYLE_STRUM_BASIC},
        {},
        1.0,
        80,
    )
    offsets_ms = [float(n.offset) * (60000 / gen.global_tempo) for n in notes]
    assert statistics.pstdev(offsets_ms) > 0



