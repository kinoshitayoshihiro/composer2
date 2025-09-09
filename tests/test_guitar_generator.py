import yaml
from music21 import chord
from music21 import instrument, harmony, articulations, pitch
from generator.guitar_generator import GuitarGenerator
from generator.guitar_generator import (
    EXEC_STYLE_BLOCK_CHORD,
    EXEC_STYLE_HAMMER_ON,
    EXEC_STYLE_PULL_OFF,
    EXEC_STYLE_STRUM_BASIC,
)


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


def test_load_external_patterns(tmp_path, monkeypatch):
    data = {"extra_pattern": {"pattern": [{"offset": 0.0, "duration": 1.0}]}}
    file = tmp_path / "patterns.yml"
    file.write_text(yaml.safe_dump(data))

    gen = GuitarGenerator(
        global_settings={},
        default_instrument=instrument.Guitar(),
        part_name="guitar",
        global_tempo=120,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
        external_patterns_path=str(file),
    )
    assert "extra_pattern" in gen.part_parameters


def test_timing_variation_jitter():
    gen = GuitarGenerator(
        global_settings={},
        default_instrument=instrument.Guitar(),
        part_name="guitar",
        global_tempo=120,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
        timing_variation=0.05,
    )
    gen.rng.seed(0)
    notes = gen._create_notes_from_event(
        harmony.ChordSymbol("C"),
        {"execution_style": "block_chord"},
        {},
        1.0,
        80,
    )
    assert notes[0].offset != 0.0


def test_custom_tuning_applied():
    gen_std = GuitarGenerator(
        global_settings={},
        default_instrument=instrument.Guitar(),
        part_name="g1",
        global_tempo=120,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
        tuning=[0, 0, 0, 0, 0, 0],
    )
    gen_drop = GuitarGenerator(
        global_settings={},
        default_instrument=instrument.Guitar(),
        part_name="g2",
        global_tempo=120,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
        tuning=[-2, 0, 0, 0, 0, 0],
    )
    p_std = gen_std._get_guitar_friendly_voicing(harmony.ChordSymbol("E"), 1)[0]
    p_drop = gen_drop._get_guitar_friendly_voicing(harmony.ChordSymbol("E"), 1)[0]
    assert int(p_drop.ps - p_std.ps) == -2


def test_tuning_preset_drop_d():
    gen = GuitarGenerator(
        global_settings={},
        default_instrument=instrument.Guitar(),
        part_name="g3",
        global_tempo=120,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
        tuning="drop_d",
    )
    assert gen.tuning == [-2, 0, 0, 0, 0, 0]
    gen_std = GuitarGenerator(
        global_settings={},
        default_instrument=instrument.Guitar(),
        part_name="g4",
        global_tempo=120,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
        tuning="standard",
    )
    p_std = gen_std._get_guitar_friendly_voicing(harmony.ChordSymbol("E"), 1)[0]
    p_drop = gen._get_guitar_friendly_voicing(harmony.ChordSymbol("E"), 1)[0]
    assert int(p_drop.ps - p_std.ps) == -2


def test_export_musicxml(tmp_path):
    gen = GuitarGenerator(
        global_settings={},
        default_instrument=instrument.Guitar(),
        part_name="guitar",
        global_tempo=120,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
    )
    gen.compose(section_data=_basic_section())
    path = tmp_path / "out.xml"
    gen.export_musicxml(str(path))
    assert path.exists() and path.stat().st_size > 0


def test_export_tab_xml_and_ascii(tmp_path):
    gen = GuitarGenerator(
        global_settings={},
        default_instrument=instrument.Guitar(),
        part_name="guitar",
        global_tempo=120,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
    )
    gen.compose(section_data=_basic_section())

    xml_path = tmp_path / "out.xml"
    gen.export_tab(str(xml_path), format="xml")
    assert xml_path.exists() and xml_path.stat().st_size > 0

    ascii_path = tmp_path / "out.txt"
    gen.export_tab(str(ascii_path), format="ascii")
    assert ascii_path.exists() and ascii_path.stat().st_size > 0


def test_gate_length_variation_range():
    gen = GuitarGenerator(
        global_settings={},
        default_instrument=instrument.Guitar(),
        part_name="guitar",
        global_tempo=120,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
        gate_length_variation=0.2,
    )
    gen.rng.seed(1)
    notes = gen._create_notes_from_event(
        harmony.ChordSymbol("C"),
        {"execution_style": EXEC_STYLE_BLOCK_CHORD},
        {},
        1.0,
        80,
    )
    dur = notes[0].quarterLength
    base = 1.0 * 0.9
    assert base * 0.8 <= dur <= base * 1.2


def test_stroke_direction_velocity():
    gen = GuitarGenerator(
        global_settings={},
        default_instrument=instrument.Guitar(),
        part_name="g",
        global_tempo=120,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
    )
    gen.default_velocity_curve = None
    cs = harmony.ChordSymbol("C")
    notes_down = gen._create_notes_from_event(
        cs,
        {"execution_style": EXEC_STYLE_BLOCK_CHORD},
        {"current_event_stroke": "down"},
        1.0,
        80,
    )
    notes_up = gen._create_notes_from_event(
        cs,
        {"execution_style": EXEC_STYLE_BLOCK_CHORD},
        {"current_event_stroke": "up"},
        1.0,
        80,
    )
    vel_down = notes_down[0].notes[0].volume.velocity if isinstance(notes_down[0], chord.Chord) else notes_down[0].volume.velocity
    vel_up = notes_up[0].notes[0].volume.velocity if isinstance(notes_up[0], chord.Chord) else notes_up[0].volume.velocity
    assert vel_down > vel_up


def test_palm_mute_shorter_sustain():
    gen = GuitarGenerator(
        global_settings={},
        default_instrument=instrument.Guitar(),
        part_name="g",
        global_tempo=120,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
    )
    cs = harmony.ChordSymbol("C")
    notes_norm = gen._create_notes_from_event(
        cs,
        {"execution_style": EXEC_STYLE_BLOCK_CHORD},
        {},
        1.0,
        80,
    )
    notes_pm = gen._create_notes_from_event(
        cs,
        {"execution_style": EXEC_STYLE_BLOCK_CHORD},
        {"palm_mute": True},
        1.0,
        80,
    )
    dur_norm = notes_norm[0].quarterLength if isinstance(notes_norm[0], chord.Chord) else notes_norm[0].quarterLength
    dur_pm = notes_pm[0].quarterLength if isinstance(notes_pm[0], chord.Chord) else notes_pm[0].quarterLength
    assert dur_pm < dur_norm


def test_strum_basic_stroke_velocity():
    gen = GuitarGenerator(
        global_settings={},
        default_instrument=instrument.Guitar(),
        part_name="g",
        global_tempo=120,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
    )
    cs = harmony.ChordSymbol("C")
    notes_down = gen._create_notes_from_event(
        cs,
        {"execution_style": EXEC_STYLE_STRUM_BASIC},
        {"current_event_stroke": "down"},
        1.0,
        80,
    )
    notes_up = gen._create_notes_from_event(
        cs,
        {"execution_style": EXEC_STYLE_STRUM_BASIC},
        {"current_event_stroke": "up"},
        1.0,
        80,
    )
    avg_down = sum(n.volume.velocity for n in notes_down) / len(notes_down)
    avg_up = sum(n.volume.velocity for n in notes_up) / len(notes_up)
    assert avg_down > avg_up


def test_internal_default_patterns():
    gen = GuitarGenerator(
        global_settings={},
        default_instrument=instrument.Guitar(),
        part_name="guitar",
        global_tempo=120,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
    )
    keys = {
        "guitar_rhythm_quarter",
        "guitar_rhythm_syncopation",
        "guitar_rhythm_shuffle",
    }
    assert keys.issubset(gen.part_parameters.keys())

    qpat = gen.part_parameters["guitar_rhythm_quarter"]["pattern"]
    assert qpat[0]["offset"] == 0.0
    assert qpat[1]["offset"] == 1.0
    assert qpat[1]["articulation"] == "palm_mute"


def test_event_articulation_staccato():
    gen = GuitarGenerator(
        global_settings={},
        default_instrument=instrument.Guitar(),
        part_name="guitar",
        global_tempo=120,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
    )

    notes = gen._create_notes_from_event(
        harmony.ChordSymbol("C"),
        {"execution_style": EXEC_STYLE_BLOCK_CHORD},
        {"articulation": "staccato"},
        1.0,
        80,
    )

    assert isinstance(notes[0], chord.Chord)
    for n in notes[0].notes:
        stacs = [a for a in n.articulations if isinstance(a, articulations.Staccato)]
        assert len(stacs) == 1


def test_pattern_level_articulation():
    gen = GuitarGenerator(
        global_settings={},
        default_instrument=instrument.Guitar(),
        part_name="guitar",
        global_tempo=120,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
    )

    notes = gen._create_notes_from_event(
        harmony.ChordSymbol("C"),
        {"execution_style": EXEC_STYLE_BLOCK_CHORD, "articulation": "staccato"},
        {},
        1.0,
        80,
    )

    assert isinstance(notes[0], chord.Chord)
    for n in notes[0].notes:
        stacs = [a for a in n.articulations if isinstance(a, articulations.Staccato)]
        assert len(stacs) == 1


def test_event_articulation_slide():
    gen = GuitarGenerator(
        global_settings={},
        default_instrument=instrument.Guitar(),
        part_name="guitar",
        global_tempo=120,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
    )

    notes = gen._create_notes_from_event(
        harmony.ChordSymbol("C"),
        {"execution_style": EXEC_STYLE_BLOCK_CHORD},
        {"articulation": "slide"},
        1.0,
        80,
    )

    assert isinstance(notes[0], chord.Chord)
    for n in notes[0].notes:
        slides = [a for a in n.articulations if isinstance(a, articulations.IndeterminateSlide)]
        assert len(slides) == 1


def test_pattern_level_accent():
    gen = GuitarGenerator(
        global_settings={},
        default_instrument=instrument.Guitar(),
        part_name="guitar",
        global_tempo=120,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
    )

    notes = gen._create_notes_from_event(
        harmony.ChordSymbol("C"),
        {"execution_style": EXEC_STYLE_BLOCK_CHORD, "articulation": "accent"},
        {},
        1.0,
        80,
    )

    assert isinstance(notes[0], chord.Chord)
    for n in notes[0].notes:
        accs = [a for a in n.articulations if isinstance(a, articulations.Accent)]
        assert len(accs) == 1


def test_multiple_articulations_list():
    gen = GuitarGenerator(
        global_settings={},
        default_instrument=instrument.Guitar(),
        part_name="guitar",
        global_tempo=120,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
    )

    notes = gen._create_notes_from_event(
        harmony.ChordSymbol("C"),
        {"execution_style": EXEC_STYLE_BLOCK_CHORD, "articulation": ["accent", "slide"]},
        {},
        1.0,
        80,
    )

    assert isinstance(notes[0], chord.Chord)
    for n in notes[0].notes:
        accs = [a for a in n.articulations if isinstance(a, articulations.Accent)]
        slides = [a for a in n.articulations if isinstance(a, articulations.IndeterminateSlide)]
        assert len(accs) == 1
        assert len(slides) == 1


def _has_fret_indication(note_obj, text):
    return any(
        isinstance(a, articulations.FretIndication) and getattr(a, "number", None) == text
        for a in note_obj.articulations
    )


def _has_articulation(note_obj, art_cls):
    return any(isinstance(a, art_cls) for a in note_obj.articulations)


def test_pattern_articulation_ghost_note():
    gen = GuitarGenerator(
        global_settings={},
        default_instrument=instrument.Guitar(),
        part_name="guitar",
        global_tempo=120,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
    )

    notes = gen._create_notes_from_event(
        harmony.ChordSymbol("C"),
        {"execution_style": EXEC_STYLE_BLOCK_CHORD, "articulation": "ghost_note"},
        {},
        1.0,
        80,
    )

    assert isinstance(notes[0], chord.Chord)
    for n in notes[0].notes:
        assert _has_fret_indication(n, "ghost note")


def test_event_articulation_ghost_note():
    gen = GuitarGenerator(
        global_settings={},
        default_instrument=instrument.Guitar(),
        part_name="guitar",
        global_tempo=120,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
    )

    notes = gen._create_notes_from_event(
        harmony.ChordSymbol("C"),
        {"execution_style": EXEC_STYLE_BLOCK_CHORD},
        {"articulation": "ghost_note"},
        1.0,
        80,
    )

    assert isinstance(notes[0], chord.Chord)
    for n in notes[0].notes:
        assert _has_fret_indication(n, "ghost note")


def test_pattern_articulation_bend():
    gen = GuitarGenerator(
        global_settings={},
        default_instrument=instrument.Guitar(),
        part_name="guitar",
        global_tempo=120,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
    )

    notes = gen._create_notes_from_event(
        harmony.ChordSymbol("C"),
        {"execution_style": EXEC_STYLE_BLOCK_CHORD, "articulation": "bend"},
        {},
        1.0,
        80,
    )

    assert isinstance(notes[0], chord.Chord)
    for n in notes[0].notes:
        bends = [a for a in n.articulations if isinstance(a, articulations.FretBend)]
        assert len(bends) == 1


def test_event_articulation_bend():
    gen = GuitarGenerator(
        global_settings={},
        default_instrument=instrument.Guitar(),
        part_name="guitar",
        global_tempo=120,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
    )

    notes = gen._create_notes_from_event(
        harmony.ChordSymbol("C"),
        {"execution_style": EXEC_STYLE_BLOCK_CHORD},
        {"articulation": "bend"},
        1.0,
        80,
    )

    assert isinstance(notes[0], chord.Chord)
    for n in notes[0].notes:
        bends = [a for a in n.articulations if isinstance(a, articulations.FretBend)]
        assert len(bends) == 1


def test_pattern_articulation_hammer_on():
    gen = GuitarGenerator(
        global_settings={},
        default_instrument=instrument.Guitar(),
        part_name="guitar",
        global_tempo=120,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
    )

    notes = gen._create_notes_from_event(
        harmony.ChordSymbol("C"),
        {"execution_style": EXEC_STYLE_BLOCK_CHORD, "articulation": "hammer_on"},
        {},
        1.0,
        80,
    )

    assert isinstance(notes[0], chord.Chord)
    for n in notes[0].notes:
        hos = [a for a in n.articulations if isinstance(a, articulations.HammerOn)]
        assert len(hos) == 1


def test_event_articulation_hammer_on():
    gen = GuitarGenerator(
        global_settings={},
        default_instrument=instrument.Guitar(),
        part_name="guitar",
        global_tempo=120,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
    )

    notes = gen._create_notes_from_event(
        harmony.ChordSymbol("C"),
        {"execution_style": EXEC_STYLE_BLOCK_CHORD},
        {"articulation": "hammer_on"},
        1.0,
        80,
    )

    assert isinstance(notes[0], chord.Chord)
    for n in notes[0].notes:
        hos = [a for a in n.articulations if isinstance(a, articulations.HammerOn)]
        assert len(hos) == 1


def test_pattern_articulation_pull_off():
    gen = GuitarGenerator(
        global_settings={},
        default_instrument=instrument.Guitar(),
        part_name="guitar",
        global_tempo=120,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
    )

    notes = gen._create_notes_from_event(
        harmony.ChordSymbol("C"),
        {"execution_style": EXEC_STYLE_BLOCK_CHORD, "articulation": "pull_off"},
        {},
        1.0,
        80,
    )

    assert isinstance(notes[0], chord.Chord)
    for n in notes[0].notes:
        pos = [a for a in n.articulations if isinstance(a, articulations.PullOff)]
        assert len(pos) == 1


def test_event_articulation_pull_off():
    gen = GuitarGenerator(
        global_settings={},
        default_instrument=instrument.Guitar(),
        part_name="guitar",
        global_tempo=120,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
    )

    notes = gen._create_notes_from_event(
        harmony.ChordSymbol("C"),
        {"execution_style": EXEC_STYLE_BLOCK_CHORD},
        {"articulation": "pull_off"},
        1.0,
        80,
    )

    assert isinstance(notes[0], chord.Chord)
    for n in notes[0].notes:
        pos = [a for a in n.articulations if isinstance(a, articulations.PullOff)]
        assert len(pos) == 1


def test_pattern_articulation_slide_in():
    gen = GuitarGenerator(
        global_settings={},
        default_instrument=instrument.Guitar(),
        part_name="guitar",
        global_tempo=120,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
    )

    notes = gen._create_notes_from_event(
        harmony.ChordSymbol("C"),
        {"execution_style": EXEC_STYLE_BLOCK_CHORD, "articulation": "slide_in"},
        {},
        1.0,
        80,
    )

    assert isinstance(notes[0], chord.Chord)
    for n in notes[0].notes:
        slides = [a for a in n.articulations if isinstance(a, articulations.IndeterminateSlide)]
        assert len(slides) == 1


def test_event_articulation_slide_in():
    gen = GuitarGenerator(
        global_settings={},
        default_instrument=instrument.Guitar(),
        part_name="guitar",
        global_tempo=120,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
    )

    notes = gen._create_notes_from_event(
        harmony.ChordSymbol("C"),
        {"execution_style": EXEC_STYLE_BLOCK_CHORD},
        {"articulation": "slide_in"},
        1.0,
        80,
    )

    assert isinstance(notes[0], chord.Chord)
    for n in notes[0].notes:
        slides = [a for a in n.articulations if isinstance(a, articulations.IndeterminateSlide)]
        assert len(slides) == 1


def test_slide_offset_params():
    gen = GuitarGenerator(
        global_settings={},
        default_instrument=instrument.Guitar(),
        part_name="guitar",
        global_tempo=120,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
    )

    notes = gen._create_notes_from_event(
        harmony.ChordSymbol("C"),
        {"execution_style": EXEC_STYLE_BLOCK_CHORD},
        {"slide_in_offset": 0.2, "slide_out_offset": 0.8},
        1.0,
        80,
    )

    assert isinstance(notes[0], chord.Chord)
    slide = [a for a in notes[0].notes[0].articulations if isinstance(a, articulations.IndeterminateSlide)][0]
    assert getattr(slide.editorial, "slide_in_offset", None) == 0.2
    assert getattr(slide.editorial, "slide_out_offset", None) == 0.8


def test_bend_params():
    gen = GuitarGenerator(
        global_settings={},
        default_instrument=instrument.Guitar(),
        part_name="guitar",
        global_tempo=120,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
    )

    notes = gen._create_notes_from_event(
        harmony.ChordSymbol("C"),
        {"execution_style": EXEC_STYLE_BLOCK_CHORD},
        {"bend_amount": 1.5, "bend_release_offset": 0.7},
        1.0,
        80,
    )

    assert isinstance(notes[0], chord.Chord)
    bend = [a for a in notes[0].notes[0].articulations if isinstance(a, articulations.FretBend)][0]
    assert getattr(bend.editorial, "bend_amount", None) == 1.5
    assert getattr(bend.editorial, "bend_release_offset", None) == 0.7


def test_merge_pattern_event_articulations():
    gen = GuitarGenerator(
        global_settings={},
        default_instrument=instrument.Guitar(),
        part_name="guitar",
        global_tempo=120,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
    )

    notes = gen._create_notes_from_event(
        harmony.ChordSymbol("C"),
        {"execution_style": EXEC_STYLE_BLOCK_CHORD, "articulation": "accent"},
        {"articulation": "staccato"},
        1.0,
        80,
    )

    assert isinstance(notes[0], chord.Chord)
    for n in notes[0].notes:
        accs = [a for a in n.articulations if isinstance(a, articulations.Accent)]
        stacs = [a for a in n.articulations if isinstance(a, articulations.Staccato)]
        assert len(accs) == 1
        assert len(stacs) == 1


def _section_with_chord(chord_label: str):
    s = _basic_section()
    s["original_chord_label"] = chord_label
    s["chord_symbol_for_voicing"] = chord_label
    return s


def test_hammer_on_pull_off_basic():
    gen = GuitarGenerator(
        global_settings={},
        default_instrument=instrument.Guitar(),
        part_name="g",
        global_tempo=120,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
        hammer_on_probability=1.0,
        pull_off_probability=1.0,
    )

    gen.part_parameters["hammer_pattern"] = {
        "execution_style": EXEC_STYLE_HAMMER_ON,
        "pattern": [{"offset": 0.0, "duration": 1.0}],
        "reference_duration_ql": 1.0,
    }
    gen.part_parameters["pull_pattern"] = {
        "execution_style": EXEC_STYLE_PULL_OFF,
        "pattern": [{"offset": 0.0, "duration": 1.0}],
        "reference_duration_ql": 1.0,
    }

    # compose() should reset previous note information
    gen._prev_note_pitch = pitch.Pitch("C3")
    gen.overrides = None
    sec1 = _section_with_chord("D")
    sec1["part_params"]["g"] = {"guitar_rhythm_key": "hammer_pattern"}
    part1 = gen.compose(section_data=sec1)
    note1 = part1.flatten().notes[0]
    if isinstance(note1, chord.Chord):
        note1 = note1.notes[0]
    assert not _has_articulation(note1, articulations.HammerOn)

    # direct _render_part call should use existing _prev_note_pitch
    gen._prev_note_pitch = pitch.Pitch("C3")
    gen.overrides = None
    sec2 = _section_with_chord("D")
    sec2["part_params"]["g"] = {"guitar_rhythm_key": "hammer_pattern"}
    part2 = gen._render_part(sec2)
    note2 = part2.flatten().notes[0]
    if isinstance(note2, chord.Chord):
        note2 = note2.notes[0]
    assert _has_articulation(note2, articulations.HammerOn)

    gen._prev_note_pitch = pitch.Pitch("E3")
    sec3 = _section_with_chord("D")
    sec3["part_params"]["g"] = {"guitar_rhythm_key": "pull_pattern"}
    part3 = gen._render_part(sec3)
    note3 = part3.flatten().notes[0]
    if isinstance(note3, chord.Chord):
        note3 = note3.notes[0]
    assert _has_articulation(note3, articulations.PullOff)


def test_hammer_on_pull_off_interval_threshold():
    gen = GuitarGenerator(
        global_settings={},
        default_instrument=instrument.Guitar(),
        part_name="g",
        global_tempo=120,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
        hammer_on_probability=1.0,
        pull_off_probability=1.0,
        hammer_on_interval=1,
        pull_off_interval=1,
    )

    gen.part_parameters["hammer_pattern"] = {
        "execution_style": EXEC_STYLE_HAMMER_ON,
        "pattern": [{"offset": 0.0, "duration": 1.0}],
        "reference_duration_ql": 1.0,
    }

    gen._prev_note_pitch = pitch.Pitch("C3")
    gen.overrides = None
    sec2 = _section_with_chord("D")
    sec2["part_params"]["g"] = {"guitar_rhythm_key": "hammer_pattern"}
    part2 = gen._render_part(sec2)
    note2 = part2.flatten().notes[0]
    if isinstance(note2, chord.Chord):
        note2 = note2.notes[0]
    assert not _has_articulation(note2, articulations.HammerOn)

    gen.hammer_on_interval = 2
    gen.part_parameters["hammer_on_interval"] = 2

    gen._prev_note_pitch = pitch.Pitch("C3")
    gen.overrides = None
    sec3 = _section_with_chord("D")
    sec3["part_params"]["g"] = {"guitar_rhythm_key": "hammer_pattern"}
    part3 = gen._render_part(sec3)
    note3 = part3.flatten().notes[0]
    if isinstance(note3, chord.Chord):
        note3 = note3.notes[0]
    assert _has_articulation(note3, articulations.HammerOn)


def test_hammer_on_pull_off_probability():
    gen = GuitarGenerator(
        global_settings={},
        default_instrument=instrument.Guitar(),
        part_name="g",
        global_tempo=120,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
        hammer_on_probability=0.0,
        pull_off_probability=0.0,
    )

    gen.part_parameters["hammer_pattern"] = {
        "execution_style": EXEC_STYLE_HAMMER_ON,
        "pattern": [{"offset": 0.0, "duration": 1.0}],
        "reference_duration_ql": 1.0,
    }

    gen._prev_note_pitch = pitch.Pitch("C3")
    gen.overrides = None
    sec2 = _section_with_chord("D")
    sec2["part_params"]["g"] = {"guitar_rhythm_key": "hammer_pattern"}
    part2 = gen._render_part(sec2)
    note2 = part2.flatten().notes[0]
    if isinstance(note2, chord.Chord):
        note2 = note2.notes[0]
    assert not _has_articulation(note2, articulations.HammerOn)

    gen.hammer_on_probability = 1.0
    gen.part_parameters["hammer_on_probability"] = 1.0

    gen._prev_note_pitch = pitch.Pitch("C3")
    gen.overrides = None
    sec3 = _section_with_chord("D")
    sec3["part_params"]["g"] = {"guitar_rhythm_key": "hammer_pattern"}
    part3 = gen._render_part(sec3)
    note3 = part3.flatten().notes[0]
    if isinstance(note3, chord.Chord):
        note3 = note3.notes[0]
    assert _has_articulation(note3, articulations.HammerOn)

