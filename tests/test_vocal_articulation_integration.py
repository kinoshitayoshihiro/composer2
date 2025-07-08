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

