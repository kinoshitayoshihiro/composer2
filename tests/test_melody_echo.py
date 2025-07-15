from music21 import instrument, pitch

from generator.piano_generator import PianoGenerator


def test_apply_melody_echo() -> None:
    gen = PianoGenerator(
        part_name="piano",
        default_instrument=instrument.Piano(),
        global_tempo=120,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
        main_cfg={},
    )
    series = [pitch.Pitch("C4"), pitch.Pitch("D4")]
    notes = gen.apply_melody_echo(series, 0.5)
    assert len(notes) == 2
    assert notes[0].offset == 0.5
