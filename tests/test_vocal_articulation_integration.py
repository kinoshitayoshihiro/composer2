import logging
from generator.vocal_generator import VocalGenerator


def _make_data():
    return [
        {"offset": 0.0, "pitch": "C4", "length": 1.0, "velocity": 90},
        {"offset": 1.0, "pitch": "D4", "length": 1.0, "velocity": 90},
    ]


def test_vibrato_engine_integration():
    gen = VocalGenerator()
    part = gen.compose(
        _make_data(), processed_chord_stream=[], humanize_opt=False, lyrics_words=["a", "b"]
    )
    cc74 = [e for e in getattr(part, "extra_cc", []) if e.get("cc") == 74]
    assert cc74, "aftertouch events missing"
    bends = getattr(part, "pitch_bends", [])
    assert bends and all("pitch" in b for b in bends)


def test_disable_articulation_suppresses_events():
    gen = VocalGenerator(enable_articulation=False)
    part = gen.compose(
        [
            {"offset": 0.0, "pitch": "C4", "length": 1.0, "velocity": 80},
            {"offset": 1.0, "pitch": "D4", "length": 1.0, "velocity": 80},
        ],
        processed_chord_stream=[],
        humanize_opt=False,
        lyrics_words=["a", "b"],
    )
    # No aftertouch CC74
    assert not any(e.get("cc") == 74 for e in getattr(part, "extra_cc", []))
    # No pitch bend events
    assert not getattr(part, "pitch_bends", [])


def test_gliss_and_trill_markers():
    data = [
        {"offset": 0.0, "pitch": "C4", "length": 1.0, "velocity": 80},
        {"offset": 1.0, "pitch": "E4", "length": 1.0, "velocity": 80},
    ]
    gen = VocalGenerator(enable_articulation=True)
    part = gen.compose(data, [], humanize_opt=False, lyrics_words=["[gliss]", "[trill]"])
    # gliss produces pitch_bends between C4→E4
    pb = getattr(part, "pitch_bends", [])
    assert any(b["pitch"] != 0 for b in pb)
    # trill produces CC74 events
    cc74 = [e for e in getattr(part, "extra_cc", []) if e.get("cc") == 74]
    assert cc74

