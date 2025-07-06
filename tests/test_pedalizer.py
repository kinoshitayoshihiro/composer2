from utilities.pedalizer import generate_pedal_cc


def test_basic_pedal_events():
    chord_stream = [
        {"chord_symbol_for_voicing": "C", "offset": 0.0},
        {"chord_symbol_for_voicing": "F", "offset": 4.0},
        {"chord_symbol_for_voicing": "G", "offset": 8.0},
        {"chord_symbol_for_voicing": "C", "offset": 12.0},
    ]
    events = generate_pedal_cc(chord_stream)
    cc64 = [e for e in events if e[1] == 64]
    assert len(cc64) == 8
    downs = sorted([e[0] for e in cc64 if e[2] == 127])
    assert downs == [0.0, 4.0, 8.0, 12.0]
