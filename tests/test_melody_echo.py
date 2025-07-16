from music21 import instrument, pitch

from generator.piano_generator import PianoGenerator


class SimplePiano(PianoGenerator):
    def _get_pattern_keys(self, musical_intent, overrides):
        return "rh", "lh"


def make_gen(main_cfg: dict | None = None) -> SimplePiano:
    patterns = {
        "rh": {"pattern": [{"offset": 0, "duration": 1.0, "type": "chord"}], "length_beats": 1.0},
        "lh": {"pattern": [{"offset": 0, "duration": 1.0, "type": "root"}], "length_beats": 1.0},
    }
    return SimplePiano(
        part_name="piano",
        part_parameters=patterns,
        default_instrument=instrument.Piano(),
        global_tempo=120,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
        main_cfg=main_cfg or {},
    )


def test_apply_melody_echo_multiple() -> None:
    gen = make_gen()
    series = [pitch.Pitch("C4"), pitch.Pitch("D4")]
    notes = gen.apply_melody_echo(series, delay_beats=0.5, echo_factor=0.5, num_echoes=3)
    assert len(notes) == 6
    offs = sorted(n.offset for n in notes[::2])
    assert offs == [0.5, 1.0, 1.5]
    vels = [n.volume.velocity for n in notes[::3]]
    assert vels[1] < vels[0]


def test_apply_melody_echo_amplify() -> None:
    gen = make_gen()
    notes = gen.apply_melody_echo([pitch.Pitch("C4")], delay_beats=1.0, echo_factor=1.2, num_echoes=2)
    assert len(notes) == 2
    assert [n.offset for n in notes] == [1.0, 2.0]
    assert notes[1].volume.velocity > notes[0].volume.velocity


def test_default_echo_count() -> None:
    gen = make_gen()
    series = [pitch.Pitch("C4"), pitch.Pitch("D4")]
    notes = gen.apply_melody_echo(series, delay_beats=0.5, echo_factor=0.9)
    assert len(notes) == 4  # default echo_count=2


def test_default_delay() -> None:
    gen = make_gen()
    notes = gen.apply_melody_echo([pitch.Pitch("C4")], echo_factor=1.0, num_echoes=1)
    assert len(notes) == 1
    assert notes[0].offset == 0.5


def test_echo_flag_controls_call(monkeypatch) -> None:
    calls: list[bool] = []

    def spy(self, *a, **k):
        calls.append(True)
        return []

    gen = make_gen({"piano": {"enable_echo": False}})
    monkeypatch.setattr(SimplePiano, "apply_melody_echo", spy)
    section = {
        "section_name": "A",
        "absolute_offset": 0.0,
        "q_length": 1.0,
        "chord_symbol_for_voicing": "C",
        "melody": [(0.0, 60, 1.0)],
    }
    gen.compose(section_data=section)
    assert not calls

    gen = make_gen({"piano": {"enable_echo": True}})
    monkeypatch.setattr(SimplePiano, "apply_melody_echo", spy)
    gen.compose(section_data=section)
    assert calls


def test_post_process_generated_part() -> None:
    section = {
        "section_name": "A",
        "absolute_offset": 0.0,
        "q_length": 1.0,
        "chord_symbol_for_voicing": "C",
        "melody": [(0.0, 60, 1.0)],
    }
    gen = make_gen({"piano": {"enable_echo": False}})
    part = gen.compose(section_data=section)
    notes_no_echo = len(part["piano_rh"].notes)

    gen = make_gen({"piano": {"enable_echo": True, "echo_delay_beats": 0.5}})
    part = gen.compose(section_data=section)
    notes_with_echo = len(part["piano_rh"].notes)

    assert notes_with_echo > notes_no_echo


def test_post_process_default_delay() -> None:
    section = {
        "section_name": "A",
        "absolute_offset": 0.0,
        "q_length": 1.0,
        "chord_symbol_for_voicing": "C",
        "melody": [(0.0, 60, 1.0)],
    }
    gen = make_gen({"piano": {"enable_echo": True}})
    part = gen.compose(section_data=section)
    rh = part["piano_rh"]
    offsets = [n.offset for n in rh.notes if n.offset > 0]
    assert 0.5 in offsets
