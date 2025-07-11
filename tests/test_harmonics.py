from music21 import articulations, harmony


def _basic_section():
    return {
        "section_name": "A",
        "q_length": 1.0,
        "humanized_duration_beats": 1.0,
        "original_chord_label": "C",
        "chord_symbol_for_voicing": "C",
        "part_params": {},
        "musical_intent": {},
        "shared_tracks": {},
    }


def test_guitar_natural_harmonic(_basic_gen):
    gen = _basic_gen(enable_harmonics=True, prob_harmonic=1.0, rng_seed=42)
    cs = harmony.ChordSymbol("C")
    notes = gen._create_notes_from_event(
        cs, {"execution_style": "block_chord"}, {}, 1.0, 80
    )
    elems = []
    for n in notes:
        if hasattr(n, "notes"):
            elems.extend(n.notes)
        else:
            elems.append(n)
    assert any(
        isinstance(a, articulations.Harmonic)
        for n in elems
        for a in n.articulations
    )


def test_strings_harmonic_expression(_strings_gen):
    gen = _strings_gen(enable_harmonics=True, prob_harmonic=1.0, rng_seed=99)
    sec = _basic_section()
    parts = gen.compose(section_data=sec)
    vio = parts["violin_i"]
    first = vio.flatten().notes[0]
    assert any(isinstance(a, articulations.Harmonic) for a in first.articulations)
