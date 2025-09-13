import tempfile
from pathlib import Path
import random
import logging
import sys
from unittest import mock
import pytest
import types
import json
from pathlib import Path

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


def test_section_profiles_override() -> None:
    pm = _dummy_pm(8.0)
    chords = [sc.ChordSpan(i*2, (i+1)*2, 0, 'maj') for i in range(4)]
    mapping = {'phrase_note':24,'phrase_velocity':100,'phrase_length_beats':0.5,
               'cycle_phrase_notes':[], 'cycle_start_bar':0, 'cycle_mode':'bar'}
    sections = [{'start_bar':0,'end_bar':2,'tag':'verse'},
                {'start_bar':2,'end_bar':4,'tag':'chorus'}]
    profiles = {
        'verse': {'phrase_pool': {'notes':[24],'weights':[1]}},
        'chorus': {'phrase_pool': {'notes':[36],'weights':[1]}, 'accent_scale':1.2}
    }
    stats = {}
    out = sc.build_sparkle_midi(pm, chords, mapping, 0.5, 'bar', 0.0, 0,
                                'flat', 120, 0.0, 0.5,
                                section_profiles=profiles, sections=sections,
                                onset_list=[0,0,0,0], stats=stats)
    verse_notes = [n for n in out.instruments[1].notes if n.start < 4.0]
    chorus_notes = [n for n in out.instruments[1].notes if n.start >=4.0]
    assert any(n.pitch == 24 for n in verse_notes)
    assert any(n.pitch == 36 for n in chorus_notes)
    assert max(n.velocity for n in chorus_notes) > max(n.velocity for n in verse_notes)


def test_style_layer_every() -> None:
    pm = _dummy_pm(8.0)
    chords = [sc.ChordSpan(i*2,(i+1)*2,0,'maj') for i in range(4)]
    mapping = {'phrase_note':24,'phrase_velocity':100,'phrase_length_beats':0.5,
               'cycle_phrase_notes':[], 'cycle_start_bar':0, 'cycle_mode':'bar'}
    stats = {}
    out = sc.build_sparkle_midi(pm, chords, mapping, 0.5, 'bar', 0.0, 0,
                                'flat', 120, 0.0, 0.5, stats=stats)
    units = [(t, stats['downbeats'][i+1] if i+1 < len(stats['downbeats']) else pm.get_end_time())
             for i, t in enumerate(stats['downbeats'])]
    picker = sc.PoolPicker([(36,1)], rng=random.Random(0))
    sc.insert_style_layer(out, 'every', units, picker, every=2, length_beats=0.5)
    phrase_inst = [inst for inst in out.instruments if inst.name == sc.PHRASE_INST_NAME][0]
    starts = [round(n.start,2) for n in phrase_inst.notes if n.pitch==36]
    assert 0.0 in starts and 4.0 in starts


def test_voicing_smooth() -> None:
    pm = _dummy_pm(8.0)
    chords = [sc.ChordSpan(0,2,0,'maj'), sc.ChordSpan(2,4,9,'min'),
              sc.ChordSpan(4,6,0,'maj'), sc.ChordSpan(6,8,9,'min')]
    base_map = {'phrase_note':24,'phrase_velocity':100,'phrase_length_beats':0.5,
               'cycle_phrase_notes':[], 'cycle_start_bar':0, 'cycle_mode':'bar',
               'chord_input_range':{'lo':48,'hi':72}}
    m1 = dict(base_map)
    m1['voicing_mode'] = 'stacked'
    out_stacked = sc.build_sparkle_midi(pm, chords, m1, 0.5, 'bar',0,0,
                                        'flat',120,0,0.5)
    m2 = dict(base_map)
    m2['voicing_mode'] = 'smooth'
    out_smooth = sc.build_sparkle_midi(pm, chords, m2, 0.5, 'bar',0,0,
                                       'flat',120,0,0.5)
    def travel(inst):
        notes = sorted(inst.instruments[0].notes, key=lambda n:n.start)
        groups = [notes[i:i+3] for i in range(0,len(notes),3)]
        total=0
        prev=None
        for g in groups:
            pitches=sorted(n.pitch for n in g)
            if prev:
                total += sum(abs(a-b) for a,b in zip(pitches, prev))
            prev=pitches
        return total
    assert travel(out_smooth) < travel(out_stacked)


def test_density_rules() -> None:
    pm = _dummy_pm(8.0)
    chords = [sc.ChordSpan(i*2,(i+1)*2,0,'maj') for i in range(4)]
    mapping = {'phrase_note':26,'phrase_velocity':100,'phrase_length_beats':0.5,
               'cycle_phrase_notes':[], 'cycle_start_bar':0, 'cycle_mode':'bar'}
    stats = {}
    out = sc.build_sparkle_midi(pm, chords, mapping,0.5,'bar',0,0,'flat',120,0,0.5,
                                onset_list=[0,4,1,1], rest_list=[0.8,0.1,0.1,0.1], stats=stats)
    notes = [stats['bar_phrase_notes'].get(i) for i in range(4)]
    assert notes[0] == 24  # high rest -> open
    assert notes[1] == 36  # dense onsets -> high
    assert notes[2] == 26  # default


def test_fill_cadence() -> None:
    pm = _dummy_pm(8.0)
    chords = [sc.ChordSpan(i*2,(i+1)*2,0,'maj') for i in range(4)]
    mapping = {'phrase_note':24,'phrase_velocity':100,'phrase_length_beats':0.5,
               'cycle_phrase_notes':[], 'cycle_start_bar':0, 'cycle_mode':'bar',
               'style_fill':34}
    stats = {}
    out = sc.build_sparkle_midi(pm, chords, mapping,0.5,'bar',0,0,'flat',120,0,0.5, stats=stats)
    units = [(t, stats['downbeats'][i+1] if i+1 < len(stats['downbeats']) else pm.get_end_time())
             for i, t in enumerate(stats['downbeats'])]
    sections = [{'start_bar':0,'end_bar':2},{'start_bar':2,'end_bar':4}]
    cnt = sc.insert_style_fill(out, 'section_end', units, mapping,
                               sections=sections, min_gap_beats=0.5)
    assert cnt == 2
    phrase_inst = [inst for inst in out.instruments if inst.name == sc.PHRASE_INST_NAME][0]
    starts = {round(n.start,2) for n in phrase_inst.notes if n.pitch==34}
    assert starts == {units[1][0], units[3][0]}


def test_swing_shapes() -> None:
    pm = _dummy_pm(4.0)
    chords = [sc.ChordSpan(0,4,0,'maj')]
    mapping = {'phrase_note':36,'phrase_velocity':100,'phrase_length_beats':0.5,
               'cycle_phrase_notes':[], 'cycle_start_bar':0, 'cycle_mode':'bar'}
    stats1 = {}
    sc.build_sparkle_midi(pm, chords, mapping,0.5,'bar',0,0,'flat',120,0.5,0.5,
                          stats=stats1, swing_shape='offbeat')
    stats2 = {}
    sc.build_sparkle_midi(pm, chords, mapping,0.5,'bar',0,0,'flat',120,0.5,0.5,
                          stats=stats2, swing_shape='even')
    pulses1 = [t for _, t in stats1['bar_pulses'][0][:3]]
    pulses2 = [t for _, t in stats2['bar_pulses'][0][:3]]
    intervals1 = [round(pulses1[i+1]-pulses1[i],3) for i in range(2)]
    intervals2 = [round(pulses2[i+1]-pulses2[i],3) for i in range(2)]
    assert intervals1 != intervals2


def test_quantize_per_beat() -> None:
    pm = _dummy_pm(4.0)
    chords = [sc.ChordSpan(0,4,0,'maj')]
    mapping = {'phrase_note':24,'phrase_velocity':100,'phrase_length_beats':0.5,
               'cycle_phrase_notes':[], 'cycle_start_bar':0, 'cycle_mode':'bar'}
    rng = random.Random(0)
    out = sc.build_sparkle_midi(pm, chords, mapping,0.5,'bar',20.0,0,'flat',120,0,0.5,
                                quantize_strength=[1.0,0.0], rng_human=rng)
    starts = [round(n.start,3) for n in out.instruments[1].notes[:4]]
    assert starts[0] % 0.25 == 0.0  # quantized
    assert starts[1] % 0.25 != 0.0  # not quantized


def test_trend_weighting() -> None:
    pm = _dummy_pm(8.0)
    chords = [sc.ChordSpan(i*2,(i+1)*2,0,'maj') for i in range(4)]
    mapping = {'phrase_note':24,'phrase_velocity':100,'phrase_length_beats':0.5,
               'cycle_phrase_notes':[], 'cycle_start_bar':0, 'cycle_mode':'bar'}
    phrase_pool = {'pool':[(24,1),(36,1)]}
    out = sc.build_sparkle_midi(pm, chords, mapping, 0.5, 'bar',0,0,
                                'flat',120,0,0.5, phrase_pool=phrase_pool,
                                onset_list=[1,2,3,4], trend_window=1, trend_th=0.0,
                                rng_pool=random.Random(0))
    last = max(n.start for n in out.instruments[1].notes)
    high = [n.pitch for n in out.instruments[1].notes if abs(n.start-last)<1e-6][0]
    assert high == 36


def test_quantize_strength() -> None:
    pm = _dummy_pm(4.0)
    chords = [sc.ChordSpan(0,4,0,'maj')]
    mapping = {'phrase_note':24,'phrase_velocity':100,'phrase_length_beats':0.5,
               'cycle_phrase_notes':[], 'cycle_start_bar':0, 'cycle_mode':'bar'}
    out = sc.build_sparkle_midi(pm, chords, mapping, 0.5, 'bar',10.0,0,
                                'flat',120,0,0.5, quantize_strength=1.0,
                                rng_human=random.Random(0))
    starts = [n.start for n in out.instruments[1].notes]
    assert all(abs((s*2)%0.5) < 1e-6 for s in starts)


def test_sections_without_guide() -> None:
    pm = _dummy_pm(8.0)
    chords = [sc.ChordSpan(i*2,(i+1)*2,0,'maj') for i in range(4)]
    mapping = {'phrase_note':24,'phrase_velocity':100,'phrase_length_beats':0.5,
               'cycle_phrase_notes':[], 'cycle_start_bar':0, 'cycle_mode':'bar'}
    sections = [{'start_bar':0,'end_bar':2,'tag':'verse'},
                {'start_bar':2,'end_bar':4,'tag':'chorus'}]
    profiles = {'chorus':{'phrase_pool':{'notes':[36],'weights':[1]}}}
    out = sc.build_sparkle_midi(pm, chords, mapping,0.5,'bar',0,0,'flat',120,0,0.5,
                                section_profiles=profiles, sections=sections,
                                onset_list=[0,0,0,0])
    high = [n.pitch for n in out.instruments[1].notes if n.start>=4.0]
    assert 36 in high


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


def test_bar_pulses_12_8_swing_12() -> None:
    if not hasattr(pretty_midi, "TimeSignature"):
        pytest.skip("pretty_midi stub lacks TimeSignature")
    pm = _pm_with_ts(12, 8, 6.0)
    chords = [sc.ChordSpan(0, 6, 0, 'maj')]
    mapping = {'phrase_note': 36, 'phrase_velocity': 100, 'phrase_length_beats': 0.5,
               'cycle_phrase_notes': [], 'cycle_start_bar': 0, 'cycle_mode': 'bar'}
    stats = {}
    sc.build_sparkle_midi(pm, chords, mapping, 0.5, 'bar', 0.0, 0, 'flat', 120, 0.0, 4/12,
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


def test_swing_clip_guard(tmp_path) -> None:
    out = tmp_path / "o.mid"
    pm = _dummy_pm()
    called = {}
    orig_pm = sc.pretty_midi.PrettyMIDI
    orig_build = sc.build_sparkle_midi

    sc.pretty_midi.PrettyMIDI = lambda path: pm

    def fake_build(*args, **kwargs):
        called['swing'] = args[9]
        return pm

    sc.build_sparkle_midi = fake_build
    try:
        with mock.patch.object(sys, 'argv', ['prog', 'in.mid', '--out', str(out), '--dry-run', '--swing', '0.999']):
            sc.main()
    finally:
        sc.build_sparkle_midi = orig_build
        sc.pretty_midi.PrettyMIDI = orig_pm
    assert called['swing'] <= 0.9


def test_cycle_token_parser() -> None:
    tokens = ["C2", "D#2", "rest", 36]
    assert [sc.parse_note_token(t) for t in tokens] == [36, 39, None, 36]


def test_merge_reset_at_bar() -> None:
    pm = _dummy_pm(4.0)
    chords = [sc.ChordSpan(0, 2, 0, 'maj'), sc.ChordSpan(2, 4, 0, 'maj')]
    mapping = {
        'phrase_note': 36,
        'phrase_velocity': 100,
        'phrase_length_beats': 1.0,
        'phrase_merge_gap': 0.1,
        'merge_reset_at': 'bar',
        'cycle_phrase_notes': [],
        'cycle_start_bar': 0,
        'cycle_mode': 'bar',
    }
    out = sc.build_sparkle_midi(pm, chords, mapping, 1.0, 'bar', 0.0, 0, 'flat', 120, 0.0, 1.0, merge_reset_at='bar')
    notes = out.instruments[1].notes
    assert len(notes) == 2
    assert any(abs(n.start - 0.0) < 1e-6 for n in notes)
    assert any(abs(n.start - 2.0) < 1e-6 for n in notes)


def _guide_pm(pattern):
    class Dummy:
        def __init__(self, pattern):
            self._length = 2.0 * len(pattern)
            inst = pretty_midi.Instrument(0)
            t = 0.0
            for dens in pattern:
                for i in range(dens):
                    inst.notes.append(pretty_midi.Note(velocity=1, pitch=60,
                                                      start=t + i * 0.1,
                                                      end=t + i * 0.1 + 0.05))
                t += 2.0
            inst.is_drum = False
            self.instruments = [inst]
            self.time_signature_changes = []

        def get_beats(self):
            step = 0.5
            n = int(self._length / step) + 1
            return [i * step for i in range(n)]

        def get_downbeats(self):
            return self.get_beats()[::4]

        def get_end_time(self):
            return self._length

    return Dummy(pattern)


def test_guide_density_switches_keys() -> None:
    pm = _dummy_pm(6.0)
    chords = [sc.ChordSpan(0, 2, 0, 'maj'), sc.ChordSpan(2, 4, 0, 'maj'), sc.ChordSpan(4, 6, 0, 'maj')]
    mapping = {'phrase_note': 36, 'phrase_velocity': 100, 'phrase_length_beats': 0.25,
               'cycle_phrase_notes': [], 'cycle_start_bar': 0, 'cycle_mode': 'bar'}
    guide = _guide_pm([0, 1, 3])
    gmap, cc, units, rest, onset, _ = sc.summarize_guide_midi(guide, 'bar', {'low': 24, 'mid': 26, 'high': 36})
    out = sc.build_sparkle_midi(pm, chords, mapping, 1.0, 'bar', 0.0, 0, 'flat', 120, 0.0, 0.5,
                                guide_notes=gmap, guide_quant='bar')
    notes = [n.pitch for n in out.instruments[1].notes]
    assert notes == [24, 26, 36]


def test_hold_respects_no_retrigger() -> None:
    pm = _dummy_pm(4.0)
    chords = [sc.ChordSpan(0, 2, 0, 'maj'), sc.ChordSpan(2, 4, 0, 'maj')]
    mapping = {'phrase_note': 36, 'phrase_velocity': 100, 'phrase_length_beats': 0.25,
               'cycle_phrase_notes': [], 'cycle_start_bar': 0, 'cycle_mode': 'bar'}
    guide = _guide_pm([0, 0])
    gmap, cc, units, rest, onset, _ = sc.summarize_guide_midi(guide, 'bar', {'low': 24, 'mid': 26, 'high': 36})
    out = sc.build_sparkle_midi(pm, chords, mapping, 1.0, 'bar', 0.0, 0, 'flat', 120, 0.0, 0.5,
                                guide_notes=gmap, guide_quant='bar')
    notes = [n for n in out.instruments[1].notes if n.pitch == 24]
    assert len(notes) == 1


def test_auto_fill_once() -> None:
    pm = _dummy_pm(6.0)
    chords = [sc.ChordSpan(0, 2, 0, 'maj'), sc.ChordSpan(2, 4, 0, 'maj'), sc.ChordSpan(4, 6, 0, 'maj')]
    mapping = {'phrase_note': 36, 'phrase_velocity': 100, 'phrase_length_beats': 0.25,
               'cycle_phrase_notes': [], 'cycle_start_bar': 0, 'cycle_mode': 'bar'}
    guide = _guide_pm([0, 1, 0])
    gmap, cc, units, rest, onset, _ = sc.summarize_guide_midi(guide, 'bar', {'low': 24, 'mid': 26, 'high': 36})
    out = sc.build_sparkle_midi(pm, chords, mapping, 1.0, 'bar', 0.0, 0, 'flat', 120, 0.0, 0.5,
                                guide_notes=gmap, guide_quant='bar')
    cnt = sc.insert_style_fill(out, 'section_end', units, mapping,
                               sections=[{"start_bar":0,"end_bar":3}],
                               bpm=120.0)
    fill_pitch = int(mapping.get('style_fill', 34))
    notes = [n for n in out.instruments[1].notes if n.pitch == fill_pitch]
    assert cnt == 1
    assert len(notes) == 1


def test_damp_cc_range() -> None:
    guide = _guide_pm([0, 4])
    gmap, cc, units, rest, onset, _ = sc.summarize_guide_midi(guide, 'bar', {'low': 24, 'mid': 26, 'high': 36})
    vals = [v for _, v in cc]
    assert min(vals) >= 0 and max(vals) <= 127
    assert vals[0] > vals[1]


def test_rest_silence_threshold() -> None:
    guide = _guide_pm([0, 1])
    gmap, cc, units, rest, onset, _ = sc.summarize_guide_midi(guide, 'bar',
                                                          {'low': 24, 'mid': 26, 'high': 36},
                                                          rest_silence_th=0.8)
    assert 0 not in gmap


def test_auto_fill_long_rest() -> None:
    pm = _dummy_pm(4.0)
    chords = [sc.ChordSpan(0, 2, 0, 'maj'), sc.ChordSpan(2, 4, 0, 'maj')]
    mapping = {'phrase_note': 36, 'phrase_velocity': 100, 'phrase_length_beats': 0.25,
               'cycle_phrase_notes': [], 'cycle_start_bar': 0, 'cycle_mode': 'bar'}
    guide = _guide_pm([1, 0])
    gmap, cc, units, rest, onset, _ = sc.summarize_guide_midi(guide, 'bar', {'low': 24, 'mid': 26, 'high': 36})
    out = sc.build_sparkle_midi(pm, chords, mapping, 1.0, 'bar', 0.0, 0, 'flat', 120, 0.0, 0.5,
                                guide_notes=gmap, guide_quant='bar')
    cnt = sc.insert_style_fill(out, 'long_rest', units, mapping,
                               rest_ratio_list=rest, rest_th=0.8, bpm=120.0)
    fill_pitch = int(mapping.get('style_fill', 34))
    notes = [n for n in out.instruments[1].notes if n.pitch == fill_pitch]
    assert cnt == 1
    assert notes and abs(notes[0].start - units[0][0]) < 1e-6


def test_damp_curve_and_smooth() -> None:
    guide = _guide_pm([0, 4, 0])
    _, cc_lin, units, rest, onset, _ = sc.summarize_guide_midi(guide, 'bar', {'low': 24, 'mid': 26, 'high': 36})
    _, cc_exp, _, _, _, _ = sc.summarize_guide_midi(guide, 'bar', {'low': 24, 'mid': 26, 'high': 36},
                                                curve='exp', gamma=2.0)
    vals_lin = [v for _, v in cc_lin]
    vals_exp = [v for _, v in cc_exp]
    assert vals_exp[1] < vals_lin[1]
    _, cc_smooth, _, _, _, _ = sc.summarize_guide_midi(guide, 'bar', {'low': 24, 'mid': 26, 'high': 36},
                                                   smooth_sigma=1.0)
    vals_smooth = [v for _, v in cc_smooth]
    assert vals_smooth[1] > vals_lin[1]


def test_threshold_note_tokens() -> None:
    guide = _guide_pm([0, 1, 3])
    gmap, cc, units, rest, onset, _ = sc.summarize_guide_midi(
        guide, 'bar', {'low': 'C1', 'mid': 'D1', 'high': 'C2'})
    assert gmap[0] == sc.parse_midi_note('C1')
    assert gmap[1] == sc.parse_midi_note('D1')
    assert gmap[2] == sc.parse_midi_note('C2')


def test_phrase_pool_weighted_seed() -> None:
    pm = _dummy_pm(8.0)
    chords = [sc.ChordSpan(i * 2, (i + 1) * 2, 0, 'maj') for i in range(4)]
    mapping = {'phrase_note': 36, 'phrase_velocity': 100, 'phrase_length_beats': 0.25,
               'cycle_phrase_notes': [], 'cycle_start_bar': 0, 'cycle_mode': 'bar',
               'phrase_hold': 'bar', 'phrase_merge_gap': -1.0}
    pool = [(24, 1.0), (26, 3.0)]
    random.seed(1)
    stats = {}
    sc.build_sparkle_midi(pm, chords, mapping, 1.0, 'bar', 0.0, 0, 'flat', 120, 0.0, 0.5,
                          phrase_pool=pool, phrase_pick='weighted', stats=stats)
    seq = [stats['bar_phrase_notes'][i] for i in range(4)]
    assert seq == [24, 26, 24, 26]


def test_cc_thinning() -> None:
    events = [(0.0, 10), (0.3, 12), (0.8, 20), (1.0, 21)]
    th = sc.thin_cc_events(events, min_interval_beats=0.5, deadband=2, clip=(8, 15))
    assert len(th) < len(events)
    assert all(8 <= v <= 15 for _, v in th)


def test_fill_gap_avoid() -> None:
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0, name=sc.PHRASE_INST_NAME)
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=36, start=0.0, end=0.1))
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=40, start=3.0, end=3.1))
    pm.instruments.append(inst)
    units = [(1.0, 2.0)]
    mapping = {'phrase_velocity': 100}
    cnt = sc.insert_style_fill(pm, 'section_end', units, mapping,
                               sections=[{'start_bar': 0, 'end_bar': 1}],
                               bpm=120.0, min_gap_beats=3.0, avoid_pitches={36})
    assert cnt == 0
    assert len(inst.notes) == 2


def test_phrase_change_lead() -> None:
    pm = _dummy_pm(4.0)
    chords = [sc.ChordSpan(0, 2, 0, 'maj'), sc.ChordSpan(2, 4, 0, 'maj')]
    mapping = {'phrase_velocity': 100, 'phrase_length_beats': 0.25,
               'cycle_phrase_notes': [36, 37], 'cycle_mode': 'bar',
               'phrase_hold': 'bar'}
    out = sc.build_sparkle_midi(pm, chords, mapping, 1.0, 'bar', 0.0, 0,
                                'flat', 120, 0.0, 0.5,
                                phrase_change_lead_beats=0.5)
    inst = out.instruments[1]
    assert any(abs(n.start - 1.75) < 1e-6 and n.pitch == 37 for n in inst.notes)


def test_rest_silence_hold_off() -> None:
    pm = _dummy_pm(4.0)
    chords = [sc.ChordSpan(0, 4, 0, 'maj')]
    mapping = {'phrase_note': 36, 'phrase_velocity': 100,
               'phrase_length_beats': 0.25, 'cycle_phrase_notes': [],
               'cycle_mode': 'bar', 'phrase_hold': 'chord'}
    guide = _guide_pm([1, 0])
    gmap, cc, units, rest, onset, _ = sc.summarize_guide_midi(
        guide, 'bar', {'low': 36, 'mid': 36, 'high': 36},
        rest_silence_th=1.0)
    out = sc.build_sparkle_midi(pm, chords, mapping, 1.0, 'bar', 0.0, 0,
                                'flat', 120, 0.0, 0.5,
                                guide_notes=gmap, guide_quant='bar',
                                guide_units=[(0.0, 4.0), (4.0, 8.0)],
                                rest_silence_hold_off=True)
    inst = out.instruments[1]
    assert len(inst.notes) == 1
    assert abs(inst.notes[0].end - 2.0) < 1e-6


def test_stop_key_on_rest() -> None:
    pm = _dummy_pm(4.0)
    chords = [sc.ChordSpan(0, 4, 0, 'maj')]
    mapping = {'phrase_note': 36, 'phrase_velocity': 100,
               'phrase_length_beats': 0.25, 'cycle_phrase_notes': [],
               'cycle_mode': 'bar', 'style_stop': 41}
    guide_notes = {0: 24}  # second unit rest
    guide_units = [(0.0, 1.0), (1.0, 2.0)]
    out = sc.build_sparkle_midi(pm, chords, mapping, 1.0, 'bar', 0.0, 0,
                                'flat', 120, 0.0, 0.5,
                                guide_notes=guide_notes, guide_quant='bar',
                                guide_units=guide_units,
                                rest_silence_send_stop=True,
                                stop_min_gap_beats=1.0,
                                stop_velocity=80)
    inst = out.instruments[1]
    stops = [n for n in inst.notes if n.pitch == 41]
    assert len(stops) == 1
    assert abs(stops[0].start - 0.5) < 1e-6


def test_guide_thresholds_list_roundrobin() -> None:
    guide = _guide_pm([1, 1, 1])
    thresholds = {'low': 24, 'mid': [["D1", 1.0], ["E1", 1.0]], 'high': 36}
    gmap, _, _, _, _, _ = sc.summarize_guide_midi(guide, 'bar', thresholds, pick_mode='roundrobin')
    seq = [gmap[i] for i in range(3)]
    assert seq == [sc.parse_midi_note('D1'), sc.parse_midi_note('E1'), sc.parse_midi_note('D1')]


def test_phrase_pool_markov() -> None:
    pm = _dummy_pm(8.0)
    chords = [sc.ChordSpan(i * 2, (i + 1) * 2, 0, 'maj') for i in range(4)]
    mapping = {'phrase_note': 36, 'phrase_velocity': 100, 'phrase_length_beats': 0.25,
               'cycle_phrase_notes': [], 'cycle_mode': 'bar', 'phrase_hold': 'bar'}
    cfg = {'notes': [24, 26], 'T': [[0, 1], [1, 0]]}
    stats = {}
    sc.build_sparkle_midi(pm, chords, mapping, 1.0, 'bar', 0.0, 0, 'flat', 120, 0.0, 0.5,
                          phrase_pool=sc.parse_phrase_pool_arg(json.dumps(cfg)),
                          phrase_pick='markov', stats=stats)
    seq = [stats['bar_phrase_notes'][i] for i in range(4)]
    assert seq == [24, 26, 24, 26]


def test_accent_map() -> None:
    pm = _dummy_pm(4.0)
    pm.time_signature_changes = [types.SimpleNamespace(numerator=4, denominator=4, time=0.0),
                                 types.SimpleNamespace(numerator=3, denominator=4, time=2.0)]
    chords = [sc.ChordSpan(0, 2, 0, 'maj'), sc.ChordSpan(2, 4, 0, 'maj')]
    mapping = {'phrase_note': 36, 'phrase_velocity': 100, 'phrase_length_beats': 1.0,
               'cycle_phrase_notes': [], 'cycle_mode': 'bar',
               'accent_map': {'4/4': [1.0, 0.5, 1.0, 0.5], '3/4': [0.2, 0.2, 1.0]}}
    stats = {}
    sc.build_sparkle_midi(pm, chords, mapping, 1.0, 'bar', 0.0, 0, 'flat', 120, 0.0, 0.5,
                          accent_map=mapping['accent_map'], stats=stats)
    v1 = stats['bar_velocities'][0][0]
    v2 = stats['bar_velocities'][1][0]
    assert v1 > v2


def test_no_repeat_window_limit() -> None:
    pool = [(24, 1.0), (26, 0.1)]
    picker = sc.PoolPicker(pool, mode='weighted', no_repeat_window=2, rng=random.Random(0))
    seq = [picker.pick() for _ in range(10)]
    assert all(seq[i] != seq[i - 1] or seq[i] != seq[i - 2] for i in range(2, len(seq)))

