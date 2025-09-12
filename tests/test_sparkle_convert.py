import tempfile
from pathlib import Path
import random
import logging
import sys
from unittest import mock
import pytest
import types

try:
    import pretty_midi  # type: ignore
except Exception:  # pragma: no cover
    from tests._stubs import pretty_midi  # type: ignore

from ujam import sparkle_convert as sc


def _dummy_pm(length: float = 6.0):
    class Dummy:
        def __init__(self, length: float) -> None:
            self._length = length
            inst = pretty_midi.Instrument(0)
            inst.notes.append(pretty_midi.Note(velocity=1, pitch=60, start=0.0, end=length))
            inst.is_drum = False
            self.instruments = [inst]
            self.time_signature_changes = []

        def get_beats(self):
            step = 0.5  # 120bpm
            n = int(self._length / step) + 1
            return [i * step for i in range(n)]

        def get_downbeats(self):
            return self.get_beats()[::4]

        def get_end_time(self):
            return self._length

        def get_tempo_changes(self):
             return [0.0], [120.0]

    return Dummy(length)


def test_parse_midi_note_enharmonic() -> None:
    assert sc.parse_midi_note("E#4") == 65
    assert sc.parse_midi_note("Cb3") == 59
    assert sc.parse_midi_note("Ｆ＃2") == 42
    assert sc.parse_midi_note("Ｃ３") == 48


def test_place_in_range() -> None:
    assert sc.place_in_range([40, 44, 47], 45, 60) == [52, 56, 59]


def test_place_in_range_closed() -> None:
    assert sc.place_in_range([60, 64, 67], 50, 64, voicing_mode='closed') == [55, 60, 64]


def test_cycle_mode_bar_rest() -> None:
    pm = _dummy_pm()
    chords = [sc.ChordSpan(0, 2, 0, 'maj'), sc.ChordSpan(2, 4, 0, 'maj'), sc.ChordSpan(4, 6, 0, 'maj')]
    mapping = {
        'phrase_note': 36,
        'phrase_velocity': 100,
        'phrase_length_beats': 0.25,
        'cycle_phrase_notes': [24, None, 26],
        'cycle_start_bar': 0,
        'cycle_mode': 'bar'
    }
    out = sc.build_sparkle_midi(pm, chords, mapping, 1.0, 'bar', 0.0, 0, 'flat', 120, 0.0, 0.5)
    notes = out.instruments[1].notes
    assert any(n.pitch == 24 for n in notes if n.start < 2)
    assert not any(2 <= n.start < 4 for n in notes)
    assert any(n.pitch == 26 for n in notes if 4 <= n.start < 6)


def test_cycle_mode_chord() -> None:
    pm = _dummy_pm()
    chords = [sc.ChordSpan(0, 2, 0, 'maj'), sc.ChordSpan(2, 4, 0, 'maj'), sc.ChordSpan(4, 6, 0, 'maj')]
    mapping = {
        'phrase_note': 36,
        'phrase_velocity': 100,
        'phrase_length_beats': 0.25,
        'cycle_phrase_notes': [24, 26],
        'cycle_start_bar': 0,
        'cycle_mode': 'chord'
    }
    out = sc.build_sparkle_midi(pm, chords, mapping, 1.0, 'chord', 0.0, 0, 'flat', 120, 0.0, 0.5)
    notes = out.instruments[1].notes
    assert any(n.pitch == 24 for n in notes if n.start < 2)
    assert any(n.pitch == 26 for n in notes if 2 <= n.start < 4)


def test_write_template_path() -> None:
    content = sc.generate_mapping_template(True)
    with tempfile.NamedTemporaryFile(delete=False) as fp:
        Path(fp.name).write_text(content)
        assert 'cycle_mode' in Path(fp.name).read_text()


def test_humanize_seed_repro() -> None:
    pm = _dummy_pm()
    chords = [sc.ChordSpan(0, 2, 0, 'maj')]
    mapping = {
        'phrase_note': 36,
        'phrase_velocity': 100,
        'phrase_length_beats': 0.25,
        'cycle_phrase_notes': [],
        'cycle_start_bar': 0,
        'cycle_mode': 'bar'
    }
    random.seed(123)
    out1 = sc.build_sparkle_midi(pm, chords, mapping, 1.0, 'bar', 10.0, 4, 'flat', 120, 0.0, 0.5)
    random.seed(123)
    out2 = sc.build_sparkle_midi(pm, chords, mapping, 1.0, 'bar', 10.0, 4, 'flat', 120, 0.0, 0.5)
    n1 = [(round(n.start, 4), round(n.end, 4), n.velocity) for n in out1.instruments[1].notes]
    n2 = [(round(n.start, 4), round(n.end, 4), n.velocity) for n in out2.instruments[1].notes]
    assert n1 == n2


def _pm_with_ts(num: int, den: int, length: float = 6.0):
    pm = pretty_midi.PrettyMIDI()
    pm.time_signature_changes.append(pretty_midi.TimeSignature(num, den, 0))
    inst = pretty_midi.Instrument(0)
    inst.notes.append(pretty_midi.Note(velocity=1, pitch=60, start=0.0, end=length))
    pm.instruments.append(inst)
    return pm


def test_read_chords_yaml_ok(tmp_path) -> None:
    pytest.importorskip("yaml")
    data = "- {start: 0.0, end: 1.0, root: C, quality: maj}\n"
    p = tmp_path / "c.yaml"
    p.write_text(data)
    spans = sc.read_chords_yaml(p)
    assert spans[0].root_pc == 0


def test_read_chords_yaml_bad(tmp_path) -> None:
    pytest.importorskip("yaml")
    p = tmp_path / "c.yaml"
    p.write_text("- {start:0,end:1,quality:maj}\n")
    with pytest.raises(KeyError):
        sc.read_chords_yaml(p)
    p.write_text("- {start:0,end:1,root:H,quality:maj}\n")
    with pytest.raises(ValueError):
        sc.read_chords_yaml(p)


def test_bar_width_12_8() -> None:
    if not hasattr(pretty_midi, "TimeSignature"):
        pytest.skip("pretty_midi stub lacks TimeSignature")
    pm = _pm_with_ts(12, 8, 6.0)
    chords = [sc.ChordSpan(0, 6, 0, 'maj')]
    mapping = {'phrase_note': 36, 'phrase_velocity': 100, 'phrase_length_beats': 0.5,
               'cycle_phrase_notes': [], 'cycle_start_bar': 0, 'cycle_mode': 'bar'}
    stats = {}
    sc.build_sparkle_midi(pm, chords, mapping, 0.5, 'bar', 0.0, 0, 'flat', 120, 0.0, 0.5,
                          stats=stats)
    assert len(stats['bar_pulses'][0]) == 12


def test_cycle_start_bar_negative() -> None:
    pm = _dummy_pm()
    chords = [sc.ChordSpan(0, 2, 0, 'maj'), sc.ChordSpan(2, 4, 0, 'maj')]
    mapping = {'phrase_note': 36, 'phrase_velocity': 100, 'phrase_length_beats': 0.25,
               'cycle_phrase_notes': [24, 26], 'cycle_start_bar': -1, 'cycle_mode': 'bar'}
    out = sc.build_sparkle_midi(pm, chords, mapping, 1.0, 'bar', 0.0, 0, 'flat', 120, 0.0, 0.5)
    notes = out.instruments[1].notes
    assert any(n.pitch == 26 for n in notes if n.start < 2)
    assert any(n.pitch == 24 for n in notes if 2 <= n.start < 4)


def test_accent_validation() -> None:
    assert sc.parse_accent_arg("[1,0.8]") == [1.0, 0.8]
    assert sc.parse_accent_arg("[]") is None
    with pytest.raises(SystemExit):
        sc.parse_accent_arg("bad")
    with pytest.raises(SystemExit):
        sc.parse_accent_arg('[1, "a"]')


def test_top_note_max_strict() -> None:
    pm = _dummy_pm()
    chords = [sc.ChordSpan(0, 2, 0, 'maj')]
    mapping = {'phrase_note': 36, 'phrase_velocity': 100, 'phrase_length_beats': 0.25,
               'cycle_phrase_notes': [], 'cycle_start_bar': 0, 'cycle_mode': 'bar',
               'top_note_max': 5, 'strict': True}
    with pytest.raises(SystemExit):
        sc.build_sparkle_midi(pm, chords, mapping, 1.0, 'bar', 0.0, 0, 'flat', 120, 0.0, 0.5)


def test_cycle_stride() -> None:
    pm = _dummy_pm(8.0)
    chords = [sc.ChordSpan(i * 2, (i + 1) * 2, 0, 'maj') for i in range(4)]
    mapping = {
        'phrase_note': 36,
        'phrase_velocity': 100,
        'phrase_length_beats': 0.25,
        'cycle_phrase_notes': [24, 26],
        'cycle_start_bar': 0,
        'cycle_mode': 'bar'
    }
    out = sc.build_sparkle_midi(pm, chords, mapping, 1.0, 'bar', 0.0, 0, 'flat', 120, 0.0, 0.5, cycle_stride=2)
    notes = out.instruments[1].notes
    assert all(n.pitch == 24 for n in notes if n.start < 4)
    assert all(n.pitch == 26 for n in notes if n.start >= 4)


def test_accent_profile() -> None:
    pm = _dummy_pm(4.0)
    chords = [sc.ChordSpan(0, 4, 0, 'maj')]
    mapping = {
        'phrase_note': 36,
        'phrase_velocity': 100,
        'phrase_length_beats': 0.25,
        'cycle_phrase_notes': [],
        'cycle_start_bar': 0,
        'cycle_mode': 'bar'
    }
    out = sc.build_sparkle_midi(pm, chords, mapping, 1.0, 'bar', 0.0, 0, 'flat', 120, 0.0, 0.5, accent=[1.0, 0.5])
    vels = [n.velocity for n in out.instruments[1].notes[:4]]
    assert vels[0] == 100 and vels[1] == 50 and vels[2] == 100 and vels[3] == 50


def test_skip_phrase_in_rests() -> None:
    pm = _dummy_pm(4.0)
    chords = [sc.ChordSpan(0, 2, 0, 'maj'), sc.ChordSpan(2, 4, 0, 'rest')]
    mapping = {
        'phrase_note': 36,
        'phrase_velocity': 100,
        'phrase_length_beats': 0.25,
        'cycle_phrase_notes': [],
        'cycle_start_bar': 0,
        'cycle_mode': 'bar',
        'silent_qualities': ['rest']
    }
    out = sc.build_sparkle_midi(pm, chords, mapping, 1.0, 'bar', 0.0, 0, 'flat', 120, 0.0, 0.5, skip_phrase_in_rests=True)
    assert all(n.start < 2 for n in out.instruments[1].notes)


def test_phrase_chord_channels() -> None:
    pm = _dummy_pm()
    chords = [sc.ChordSpan(0, 2, 0, 'maj')]
    mapping = {
        'phrase_note': 36,
        'phrase_velocity': 100,
        'phrase_length_beats': 0.25,
        'cycle_phrase_notes': [],
        'cycle_start_bar': 0,
        'cycle_mode': 'bar'
    }
    out = sc.build_sparkle_midi(pm, chords, mapping, 1.0, 'bar', 0.0, 0, 'flat', 120, 0.0, 0.5, phrase_channel=1, chord_channel=2)
    assert out.instruments[0].midi_channel == 2
    assert out.instruments[1].midi_channel == 1


@pytest.mark.skipif(
    pretty_midi.PrettyMIDI.__module__ == "tests._stubs",
    reason="pretty_midi stub lacks persistent channel handling",
)
def test_channel_roundtrip(tmp_path) -> None:
    pm = _dummy_pm()
    chords = [sc.ChordSpan(0, 2, 0, 'maj')]
    mapping = {
        'phrase_note': 36,
        'phrase_velocity': 100,
        'phrase_length_beats': 0.25,
        'cycle_phrase_notes': [],
        'cycle_start_bar': 0,
        'cycle_mode': 'bar'
    }
    out = sc.build_sparkle_midi(pm, chords, mapping, 1.0, 'bar', 0.0, 0, 'flat', 120, 0.0, 0.5, phrase_channel=1, chord_channel=2)
    path = tmp_path / "round.mid"
    out.write(str(path))
    pm2 = pretty_midi.PrettyMIDI(str(path))
    assert len(pm2.instruments) == 2
    names = {inst.name for inst in pm2.instruments}
    assert "Sparkle Chords" in names
    assert any("Sparkle Phrase" in n for n in names)


def test_validate_midi_note_range() -> None:
    with pytest.raises(SystemExit):
        sc.validate_midi_note(200)


def test_dry_run_logging(tmp_path, caplog) -> None:
    out = tmp_path / "out.mid"
    orig = sc.pretty_midi.PrettyMIDI
    sc.pretty_midi.PrettyMIDI = lambda path: _dummy_pm()
    try:
        with mock.patch.object(sys, 'argv', ['prog', 'in.mid', '--out', str(out), '--dry-run']):
            with caplog.at_level(logging.INFO):
                sc.main()
        assert 'bars=' in caplog.text
        assert 'meter_map' in caplog.text
        assert not out.exists()
    finally:
        sc.pretty_midi.PrettyMIDI = orig


def test_pulse_grid_3_4_tempo_change() -> None:
    class Dummy:
        def __init__(self) -> None:
            self.instruments = []
            self.time_signature_changes = [types.SimpleNamespace(numerator=3, denominator=4, time=0.0)]

        def get_beats(self):
            return [0.0, 0.5, 1.0, 1.5, 2.5, 3.5, 4.5]

        def get_downbeats(self):
            return []

        def get_end_time(self):
            return 4.5

        def get_tempo_changes(self):
            return [0.0], [120.0]

    pm = Dummy()
    chords = [sc.ChordSpan(0, 1.5, 0, 'maj'), sc.ChordSpan(1.5, 4.5, 0, 'maj')]
    mapping = {
        'phrase_note': 36,
        'phrase_velocity': 100,
        'phrase_length_beats': 0.25,
        'cycle_phrase_notes': [],
        'cycle_start_bar': 0,
        'cycle_mode': 'bar'
    }
    out = sc.build_sparkle_midi(pm, chords, mapping, 1.0, 'bar', 0.0, 0, 'flat', 120, 0.0, 0.5)
    starts = [round(n.start, 3) for n in out.instruments[1].notes]
    assert starts == [0.0, 0.5, 1.0, 1.5, 2.5, 3.5]


def test_bar_width_6_8() -> None:
    class Dummy:
        def __init__(self) -> None:
            self.instruments = []
            self.time_signature_changes = [types.SimpleNamespace(numerator=6, denominator=8, time=0.0)]

        def get_beats(self):
            return [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]

        def get_downbeats(self):
            return []

        def get_end_time(self):
            return 4.5

        def get_tempo_changes(self):
            return [0.0], [120.0]

    pm = Dummy()
    chords = [sc.ChordSpan(0, 4.5, 0, 'maj')]
    mapping = {
        'phrase_note': 36,
        'phrase_velocity': 100,
        'phrase_length_beats': 0.25,
        'cycle_phrase_notes': [],
        'cycle_start_bar': 0,
        'cycle_mode': 'bar'
    }
    out = sc.build_sparkle_midi(pm, chords, mapping, 1.0, 'bar', 0.0, 0, 'flat', 120, 0.0, 0.5)
    starts = [round(n.start, 3) for n in out.instruments[1].notes]
    assert starts == [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]


def test_cycle_disabled_info(caplog) -> None:
    class Dummy:
        def __init__(self) -> None:
            self.instruments = []
            self.time_signature_changes = []

        def get_beats(self):
            return [0.0, 0.5]

        def get_downbeats(self):
            return []

        def get_end_time(self):
            return 1.0

        def get_tempo_changes(self):
            return [0.0], [120.0]

    pm = Dummy()
    chords = [sc.ChordSpan(0, 1.0, 0, 'maj')]
    mapping = {
        'phrase_note': 36,
        'phrase_velocity': 100,
        'phrase_length_beats': 0.25,
        'cycle_phrase_notes': [24, 26],
        'cycle_start_bar': 0,
        'cycle_mode': 'bar'
    }
    with caplog.at_level(logging.INFO):
        sc.build_sparkle_midi(pm, chords, mapping, 1.0, 'bar', 0.0, 0, 'flat', 120, 0.0, 0.5)
    assert 'cycle disabled; using fixed phrase_note=36' in caplog.text


def test_top_note_max() -> None:
    pm = _dummy_pm()
    chords = [sc.ChordSpan(0, 2, 0, 'maj')]
    mapping = {
        'phrase_note': 36,
        'phrase_velocity': 100,
        'phrase_length_beats': 0.25,
        'cycle_phrase_notes': [],
        'cycle_start_bar': 0,
        'cycle_mode': 'bar',
        'chord_octave': 5,
        'chord_input_range': {'lo': 60, 'hi': 80},
        'voicing_mode': 'stacked',
        'top_note_max': 70
    }
    stats = {}
    sc.build_sparkle_midi(pm, chords, mapping, 1.0, 'bar', 0.0, 0, 'flat', 120, 0.0, 0.5, stats=stats)
    triad = stats['triads'][0]
    assert max(triad) <= 70


def test_swing_timings() -> None:
    pm = _dummy_pm(3.0)
    chords = [sc.ChordSpan(0, 3, 0, 'maj')]
    mapping = {
        'phrase_note': 36,
        'phrase_velocity': 100,
        'phrase_length_beats': 0.25,
        'cycle_phrase_notes': [],
        'cycle_start_bar': 0,
        'cycle_mode': 'bar'
    }
    stats = {}
    out = sc.build_sparkle_midi(pm, chords, mapping, 0.5, 'bar', 0.0, 0, 'flat', 120, 0.4, 0.5, stats=stats)
    pulses = stats['bar_pulses'][0]
    diff1 = pulses[1][0] - pulses[0][0]
    diff2 = pulses[2][0] - pulses[1][0]
    assert round(diff1, 2) == 0.7 and round(diff2, 2) == 0.3


def test_phrase_hold_chord_merges_pulses() -> None:
    pm = _dummy_pm()
    chords = [sc.ChordSpan(0, 2, 0, 'maj'), sc.ChordSpan(2, 4, 0, 'maj')]
    mapping = {
        'phrase_note': 36,
        'phrase_velocity': 100,
        'phrase_length_beats': 0.25,
        'cycle_phrase_notes': [],
        'cycle_start_bar': 0,
        'cycle_mode': 'bar'
    }
    off = sc.build_sparkle_midi(pm, chords, mapping, 1.0, 'bar', 0.0, 0, 'flat', 120, 0.0, 0.5)
    mapping_hold = dict(mapping)
    mapping_hold.update({'phrase_hold': 'chord', 'phrase_merge_gap': 0.03})
    hold = sc.build_sparkle_midi(pm, chords, mapping_hold, 1.0, 'bar', 0.0, 0, 'flat', 120, 0.0, 0.5)
    assert len(hold.instruments[1].notes) < len(off.instruments[1].notes)


def test_phrase_release_and_minlen() -> None:
    pm = _dummy_pm()
    chords = [sc.ChordSpan(0, 2, 0, 'maj')]
    mapping = {
        'phrase_note': 36,
        'phrase_velocity': 100,
        'phrase_length_beats': 0.25,
        'cycle_phrase_notes': [],
        'cycle_start_bar': 0,
        'cycle_mode': 'bar',
        'phrase_release_ms': 50.0,
        'min_phrase_len_ms': 100.0
    }
    out = sc.build_sparkle_midi(pm, chords, mapping, 1.0, 'bar', 0.0, 0, 'flat', 120, 0.0, 0.5)
    note = out.instruments[1].notes[0]
    length = note.end - note.start
    assert length >= 0.1 - sc.EPS and length < 0.125


def test_held_vel_mode_max() -> None:
    pm = _dummy_pm()
    chords = [sc.ChordSpan(0, 4, 0, 'maj')]
    mapping_first = {
        'phrase_note': 36,
        'phrase_velocity': 100,
        'phrase_length_beats': 0.25,
        'cycle_phrase_notes': [],
        'cycle_start_bar': 0,
        'cycle_mode': 'bar',
        'phrase_hold': 'bar'
    }
    mapping_max = dict(mapping_first)
    mapping_max['held_vel_mode'] = 'max'
    accent = [0.5, 1.0]
    out_first = sc.build_sparkle_midi(pm, chords, mapping_first, 1.0, 'bar', 0.0, 0, 'flat', 120, 0.0, 0.5, accent=accent)
    out_max = sc.build_sparkle_midi(pm, chords, mapping_max, 1.0, 'bar', 0.0, 0, 'flat', 120, 0.0, 0.5, accent=accent)
    v1 = out_first.instruments[1].notes[0].velocity
    v2 = out_max.instruments[1].notes[0].velocity
    assert v2 > v1

