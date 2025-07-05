from music21 import stream, note
from generator.guitar_generator import GuitarGenerator


def _gen():
    from music21 import instrument
    return GuitarGenerator(
        global_settings={},
        default_instrument=instrument.Guitar(),
        part_name="g",
        global_tempo=120,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
    )


def test_style_curve_missing(monkeypatch):
    gen = _gen()
    p = stream.Part([note.Note("C4", quarterLength=1.0)])
    import utilities.style_db as sdb
    monkeypatch.delattr(sdb, "get_style_curve", raising=False)
    gen._apply_style_curve(p, "soft")
