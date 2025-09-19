import pytest

pm = pytest.importorskip("pretty_midi")

import ujam.sparkle_convert as sc


def _dummy_pm(duration: float = 8.0, *, with_base_note: bool = True) -> pm.PrettyMIDI:
    midi = pm.PrettyMIDI(initial_tempo=120)
    inst = pm.Instrument(program=0, name=sc.PHRASE_INST_NAME)
    if with_base_note:
        inst.notes.append(pm.Note(velocity=64, pitch=60, start=0.0, end=duration))
    midi.instruments.append(inst)
    midi.time_signature_changes.append(pm.TimeSignature(4, 4, 0.0))
    return midi


def _phrase_inst(midi: pm.PrettyMIDI) -> pm.Instrument:
    for inst in midi.instruments:
        if inst.name == sc.PHRASE_INST_NAME:
            return inst
    raise AssertionError("phrase instrument missing")


def test_insert_style_fill_handles_string_tokens():
    pm_out = _dummy_pm()
    units = [(i, i + 1.0) for i in range(8)]
    mapping = {
        "style_fill": "39",
        "cycle_phrase_notes": ["open_1_8", "half_mute_1_8"],
        "phrase_note": "C2",
        "phrase_note_map": {"open_1_8": 36},
    }
    sections = [
        {"start_bar": 0, "end_bar": 4, "tag": "A"},
        {"start_bar": 4, "end_bar": 8, "tag": "B"},
    ]
    count = sc.insert_style_fill(
        pm_out,
        "section_end",
        units,
        mapping,
        sections=sections,
        fill_length_beats=0.5,
        bpm=120.0,
    )
    assert isinstance(count, int)
    assert count >= 0


def test_resolve_pitch_token_note_name_and_alias():
    mapping = {
        "phrase_note_map": {"open_alias": "C3"},
    }
    assert sc._resolve_pitch_token("C4", mapping) == 60
    assert sc._resolve_pitch_token("open_alias", mapping) == 48
    assert sc._resolve_pitch_token({"pitch": 40}, mapping) == 40
    assert sc._resolve_pitch_token(40.6, mapping) == 41


def test_note_name_to_midi_unicode_accidentals():
    assert sc._note_name_to_midi("C♭3") == 47
    assert sc._note_name_to_midi("F♯2") == 42


def test_style_inject_alias_resolves_and_inserts():
    mapping = {"phrase_note_map": {"open_1_8": 36, "skip": 38}}
    cfg = sc.validate_style_inject_cfg(
        {
            "period": 2,
            "note": "open_1_8",
            "duration_beats": 0.5,
            "avoid_pitches": ["skip"],
        },
        mapping,
    )
    assert cfg["note"] == 36
    assert cfg["avoid_pitches"] == [38]
    _plan, fills, sources = sc.schedule_phrase_keys(
        4,
        [None] * 4,
        None,
        None,
        style_inject=cfg,
        fill_policy="style",
        pulse_subdiv=0.5,
    )
    assert fills
    for bar, (note, dur, _scale) in fills.items():
        assert bar % 2 == 0
        assert note == 36
        assert pytest.approx(dur) == 0.5
    for bar, reason in sources.items():
        assert bar in fills
        assert reason == "style"


def test_style_inject_accepts_numeric_string_and_alias():
    mapping = {"phrase_note_map": {"open_1_8": 36}}
    cfg = sc.validate_style_inject_cfg(
        {
            "period": 1,
            "note": "39",
            "duration_beats": 0.25,
            "avoid_pitches": ["open_1_8", "40"],
        },
        mapping,
    )
    assert cfg["note"] == 39
    assert cfg["avoid_pitches"] == [36, 40]


def test_build_avoid_set_combines_mapping_and_explicit():
    mapping = {
        "phrase_note": 36,
        "cycle_phrase_notes": ["C2", "open_1_8"],
        "phrase_token_pitch": {"open_1_8": 40},
    }
    avoid = sc.build_avoid_set(mapping, ["open_1_8", "C#2"])
    assert {36, 37, 40}.issubset(avoid)


def test_build_avoid_set_includes_style_fill_token():
    mapping = {
        "style_fill": "open_fill",
        "phrase_note_map": {"open_fill": "C4"},
    }
    avoid = sc.build_avoid_set(mapping)
    assert 60 in avoid


def test_section_label_array_normalizes_for_fills():
    pm_out = _dummy_pm(with_base_note=False)
    units = [(i, i + 1.0) for i in range(4)]
    mapping = {
        "style_fill": "40",
        "phrase_velocity": 96,
        "phrase_note": 36,
        "cycle_phrase_notes": [36],
    }
    sections = ["Verse", "Verse", "Chorus", "Chorus"]
    count = sc.insert_style_fill(
        pm_out,
        "section_end",
        units,
        mapping,
        sections=sections,
        fill_length_beats=0.5,
        bpm=120.0,
    )
    assert count == 2
    inst = _phrase_inst(pm_out)
    assert len(inst.notes) == 2
    starts = sorted(round(n.start, 3) for n in inst.notes)
    assert starts == [1.0, 3.0]


def test_insert_style_fill_fallback_uses_first_available_pitch():
    pm_out = _dummy_pm(with_base_note=False)
    units = [(i, i + 1.0) for i in range(4)]
    mapping = {
        "style_fill": "unknown_token",
        "phrase_note": 60,
        "phrase_velocity": 90,
    }
    sections = [
        {"start_bar": 0, "end_bar": 2, "tag": "A"},
        {"start_bar": 2, "end_bar": 4, "tag": "B"},
    ]
    avoid = set(range(60, 71))
    count = sc.insert_style_fill(
        pm_out,
        "section_end",
        units,
        mapping,
        sections=sections,
        fill_length_beats=0.5,
        bpm=120.0,
        avoid_pitches=avoid,
    )
    assert count >= 0
    inst = _phrase_inst(pm_out)
    if inst.notes:
        pitches = {n.pitch for n in inst.notes}
        assert 71 in pitches


def test_insert_style_fill_handles_full_avoid_set_without_hang():
    pm_out = _dummy_pm(with_base_note=False)
    units = [(i, i + 1.0) for i in range(4)]
    mapping = {"style_fill": "unknown", "phrase_note": 36}
    sections = [
        {"start_bar": 0, "end_bar": 2, "tag": "A"},
        {"start_bar": 2, "end_bar": 4, "tag": "B"},
    ]
    avoid = set(range(128))
    count = sc.insert_style_fill(
        pm_out,
        "section_end",
        units,
        mapping,
        sections=sections,
        fill_length_beats=0.5,
        bpm=120.0,
        avoid_pitches=avoid,
    )
    assert count == 0
    inst = _phrase_inst(pm_out)
    assert not inst.notes
