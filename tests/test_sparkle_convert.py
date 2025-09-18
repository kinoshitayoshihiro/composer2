import json
import logging
import random
import sys
import tempfile
import types
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

try:
    import pretty_midi  # type: ignore
except Exception:  # pragma: no cover
    from tests._stubs import pretty_midi  # type: ignore

import argparse

from ujam import sparkle_convert as sc
from ujam.consts import DAMP_INST_NAME


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

        def write(self, path: str) -> None:  # pragma: no cover
            Path(path).write_bytes(b"")

    return Dummy(length)


def test_parse_midi_note_enharmonic() -> None:
    assert sc.parse_midi_note("E#4") == 65
    assert sc.parse_midi_note("Cb3") == 59
    assert sc.parse_midi_note("Ｆ＃2") == 42
    assert sc.parse_midi_note("Ｃ３") == 48


def test_place_in_range() -> None:
    assert sc.place_in_range([40, 44, 47], 45, 60) == [52, 56, 59]


def test_place_in_range_closed() -> None:
    assert sc.place_in_range([60, 64, 67], 50, 64, voicing_mode="closed") == [55, 60, 64]


def test_normalize_sections_from_labels() -> None:
    out = sc._normalize_sections(["a", "b", "c"], 10, "sec")
    assert [
        {"start_bar": sec["start_bar"], "end_bar": sec["end_bar"]}
        for sec in out
    ] == [
        {"start_bar": 0, "end_bar": 1},
        {"start_bar": 1, "end_bar": 2},
        {"start_bar": 2, "end_bar": 10},
    ]
    assert [sec["tag"] for sec in out] == ["a", "b", "c"]


def test_normalize_sections_from_dicts() -> None:
    sections = [{"start_bar": 5, "tag": "pre"}, {"start_bar": 10, "tag": "cho"}]
    out = sc._normalize_sections(sections, 20, "sec")
    assert out == [
        {"start_bar": 5, "end_bar": 10, "tag": "pre"},
        {"start_bar": 10, "end_bar": 20, "tag": "cho"},
    ]


def test_normalize_sections_label_indices() -> None:
    out = sc._normalize_sections(["A", "B"], 8, "verse")
    assert out == [
        {"start_bar": 0, "end_bar": 1, "tag": "A"},
        {"start_bar": 1, "end_bar": 8, "tag": "B"},
    ]


def test_normalize_sections_sort_and_clamp() -> None:
    sections = [{"start_bar": 4, "tag": "B"}, {"start_bar": 0, "tag": "A"}]
    out = sc._normalize_sections(sections, 8, "sec")
    assert out == [
        {"start_bar": 0, "end_bar": 4, "tag": "A"},
        {"start_bar": 4, "end_bar": 8, "tag": "B"},
    ]


def test_normalize_sections_overlap_adjust() -> None:
    sections = [
        {"start_bar": 0, "end_bar": 4, "tag": "A"},
        {"start_bar": 3, "tag": "B"},
    ]
    out = sc._normalize_sections(sections, 12, "sec")
    assert out == [
        {"start_bar": 0, "end_bar": 4, "tag": "A"},
        {"start_bar": 4, "end_bar": 12, "tag": "B"},
    ]


def test_normalize_sections_negative_and_far() -> None:
    sections = [{"start_bar": -3, "tag": "Neg"}, {"start_bar": 999, "tag": "Far"}]
    out = sc._normalize_sections(sections, 20, "sec")
    assert out == [
        {"start_bar": 0, "end_bar": 20, "tag": "Neg"},
    ]


def test_normalize_sections_verbose_logging(caplog: pytest.LogCaptureFixture) -> None:
    sections = [
        {"start_bar": -2, "tag": "neg"},
        {"start_bar": 1, "end_bar": 1, "tag": "tight"},
    ]
    stats: Dict[str, Any] = {"_section_verbose": True}
    with caplog.at_level(logging.INFO):
        out = sc._normalize_sections(sections, 4, "sec", stats=stats)
    assert out
    messages = [rec.getMessage() for rec in caplog.records]
    assert any("section normalize input" in msg for msg in messages)
    assert any("section normalize output" in msg for msg in messages)
    assert any("sections normalized with adjustments" in msg for msg in messages)


def test_write_markers_encoding_ascii_escape() -> None:
    mapping = {"phrase_note": 36, "phrase_velocity": 96, "cycle_phrase_notes": [], "style_fill": 34}
    downbeats = [0.0, 1.0, 2.0, 3.0]
    sections = ["あ", "B", "C"]
    expected = {
        "raw": "あ",
        "ascii": "?",
        "escape": "\\u3042",
    }
    if not hasattr(pretty_midi, "Marker"):
        class _Marker:
            def __init__(self, text: str, time: float) -> None:
                self.text = text
                self.time = time

        pretty_midi.Marker = _Marker  # type: ignore[attr-defined]
    for mode, exp in expected.items():
        pm = pretty_midi.PrettyMIDI()
        inst = pretty_midi.Instrument(0, name=sc.PHRASE_INST_NAME)
        pm.instruments.append(inst)
        pm.markers = []  # type: ignore[attr-defined]
        stats: Dict[str, Any] = {}
        sc.finalize_phrase_track(
            pm,
            argparse.Namespace(auto_fill="off"),
            stats,
            mapping,
            downbeats=downbeats,
            write_markers=True,
            marker_encoding=(mode if mode != "escape" else "ESCAPE"),
            section_overrides=sections,
            section_default="verse",
        )
        assert pm.markers, mode
        assert pm.markers[0].text == exp
        assert pm.markers[1].text == "B"


def test_cycle_mode_bar_rest() -> None:
    pm = _dummy_pm()
    chords = [
        sc.ChordSpan(0, 2, 0, "maj"),
        sc.ChordSpan(2, 4, 0, "maj"),
        sc.ChordSpan(4, 6, 0, "maj"),
    ]
    mapping = {
        "phrase_note": 36,
        "phrase_velocity": 100,
        "phrase_length_beats": 0.25,
        "cycle_phrase_notes": [24, None, 26],
        "cycle_start_bar": 0,
        "cycle_mode": "bar",
    }
    out = sc.build_sparkle_midi(pm, chords, mapping, 1.0, "bar", 0.0, 0, "flat", 120, 0.0, 0.5)
    notes = out.instruments[1].notes
    assert any(n.pitch == 24 for n in notes if n.start < 2)
    assert not any(2 <= n.start < 4 for n in notes)
    assert any(n.pitch == 26 for n in notes if 4 <= n.start < 6)


def test_cycle_mode_chord() -> None:
    pm = _dummy_pm()
    chords = [
        sc.ChordSpan(0, 2, 0, "maj"),
        sc.ChordSpan(2, 4, 0, "maj"),
        sc.ChordSpan(4, 6, 0, "maj"),
    ]
    mapping = {
        "phrase_note": 36,
        "phrase_velocity": 100,
        "phrase_length_beats": 0.25,
        "cycle_phrase_notes": [24, 26],
        "cycle_start_bar": 0,
        "cycle_mode": "chord",
    }
    out = sc.build_sparkle_midi(pm, chords, mapping, 1.0, "chord", 0.0, 0, "flat", 120, 0.0, 0.5)
    notes = out.instruments[1].notes
    assert any(n.pitch == 24 for n in notes if n.start < 2)
    assert any(n.pitch == 26 for n in notes if 2 <= n.start < 4)


def test_write_template_path() -> None:
    content = sc.generate_mapping_template(True)
    with tempfile.NamedTemporaryFile(delete=False) as fp:
        Path(fp.name).write_text(content)
        assert "cycle_mode" in Path(fp.name).read_text()


def test_humanize_seed_repro() -> None:
    pm = _dummy_pm()
    chords = [sc.ChordSpan(0, 2, 0, "maj")]
    mapping = {
        "phrase_note": 36,
        "phrase_velocity": 100,
        "phrase_length_beats": 0.25,
        "cycle_phrase_notes": [],
        "cycle_start_bar": 0,
        "cycle_mode": "bar",
    }
    rng1 = random.Random(123)
    out1 = sc.build_sparkle_midi(
        pm, chords, mapping, 1.0, "bar", 10.0, 4, "flat", 120, 0.0, 0.5, rng=rng1
    )
    rng2 = random.Random(123)
    out2 = sc.build_sparkle_midi(
        pm, chords, mapping, 1.0, "bar", 10.0, 4, "flat", 120, 0.0, 0.5, rng=rng2
    )
    n1 = [(round(n.start, 4), round(n.end, 4), n.velocity) for n in out1.instruments[1].notes]
    n2 = [(round(n.start, 4), round(n.end, 4), n.velocity) for n in out2.instruments[1].notes]
    assert n1 == n2


def test_read_chords_csv_start_chord(tmp_path: Path) -> None:
    path = tmp_path / "start_chords.csv"
    path.write_text("start,chord\n0.0,C:maj\n1.5,G:min\n", encoding="utf-8")
    spans = sc.read_chords_csv(path, bpm_hint=120.0)
    assert len(spans) == 2
    assert spans[0].start == pytest.approx(0.0)
    assert spans[0].end == pytest.approx(1.5)
    assert spans[0].root_pc == sc.PITCH_CLASS["C"]
    assert spans[0].quality == "maj"
    assert spans[1].start == pytest.approx(1.5)
    assert spans[1].end == pytest.approx(3.5)  # final span extends by one bar
    assert spans[1].quality == "min"


def test_read_chords_csv_bar_chord(tmp_path: Path) -> None:
    path = tmp_path / "bar_chords.csv"
    path.write_text("bar,chord\n0,C:maj\n2,D:maj\n", encoding="utf-8")
    spans = sc.read_chords_csv(path, bpm_hint=120.0, default_ts=(4, 4))
    assert [round(s.start, 3) for s in spans] == [0.0, 4.0]
    assert [round(s.end, 3) for s in spans] == [4.0, 6.0]
    assert [s.root_pc for s in spans] == [sc.PITCH_CLASS["C"], sc.PITCH_CLASS["D"]]


def test_read_chords_csv_headerless_bars(tmp_path: Path) -> None:
    path = tmp_path / "headerless_bars.csv"
    path.write_text("0,C:maj\n2,D:maj\n", encoding="utf-8")
    spans = sc.read_chords_csv(path, bpm_hint=110.0, default_ts=(4, 4))
    sec_per_bar = (60.0 / 110.0) * 4.0
    assert [round(s.start, 4) for s in spans] == [0.0, round(2 * sec_per_bar, 4)]
    assert [round(s.end, 4) for s in spans] == [round(2 * sec_per_bar, 4), round(2 * sec_per_bar + sec_per_bar, 4)]


def test_read_chords_csv_headerless_seconds(tmp_path: Path) -> None:
    path = tmp_path / "headerless_secs.csv"
    path.write_text("0.5,G:maj\n1.25,A:min\n", encoding="utf-8")
    spans = sc.read_chords_csv(path, bpm_hint=90.0, default_ts=(4, 4))
    sec_per_bar = (60.0 / 90.0) * 4.0
    assert spans[0].start == pytest.approx(0.5)
    assert spans[0].end == pytest.approx(1.25)
    assert spans[1].start == pytest.approx(1.25)
    assert spans[1].end == pytest.approx(1.25 + sec_per_bar)
    assert [s.quality for s in spans] == ["maj", "min"]


def test_read_chords_csv_full_header(tmp_path: Path) -> None:
    path = tmp_path / "full_header.csv"
    path.write_text(
        "start,end,root,quality\n0.0,1.5,C,maj\n1.5,3.0,G,min\n",
        encoding="utf-8",
    )
    spans = sc.read_chords_csv(path, bpm_hint=100.0)
    assert [(s.start, s.end) for s in spans] == [(0.0, 1.5), (1.5, 3.0)]
    assert [s.root_pc for s in spans] == [sc.PITCH_CLASS["C"], sc.PITCH_CLASS["G"]]
    assert [s.quality for s in spans] == ["maj", "min"]


def test_read_chords_csv_meter_map(tmp_path: Path) -> None:
    path = tmp_path / "meter_map.csv"
    path.write_text("bar,chord\n0,C:maj\n1,G:maj\n2,F:maj\n", encoding="utf-8")
    meter_map = [(0.0, 3, 4), (3.0, 4, 4)]
    spans = sc.read_chords_csv(path, meter_map=meter_map, bpm_hint=120.0)
    assert [s.start for s in spans] == pytest.approx([0.0, 1.5, 3.0], rel=1e-6)
    assert [s.end for s in spans] == pytest.approx([1.5, 3.0, 5.0], rel=1e-6)


def test_read_chords_csv_invalid_symbol(tmp_path: Path) -> None:
    path = tmp_path / "invalid.csv"
    path.write_text("start,chord\n0.0,X\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        sc.read_chords_csv(path)


# --- Tests (conflict resolved): merge both branches with compatibility/skip guards ---
# Place the following into your test module (e.g., tests/test_sparkle_convert.py).
# It merges tests from both branches and uses introspection to skip when a feature
# is not present in the current API (so your suite remains green during refactors).

import inspect

import pretty_midi

import sparkle_convert as sc  # module under test

# Helpers to detect optional API surface
_DEF_SIG = None
try:
    _DEF_SIG = inspect.signature(sc.schedule_phrase_keys)
except Exception:
    _DEF_SIG = None

_HAS_MARKOV = bool(_DEF_SIG and 'markov' in _DEF_SIG.parameters)
_HAS_SECTION_POOL_WEIGHTS = bool(_DEF_SIG and 'section_pool_weights' in _DEF_SIG.parameters)
_HAS_BAR_QUALITIES = bool(_DEF_SIG and 'bar_qualities' in _DEF_SIG.parameters)
_HAS_STYLE_INJECT = bool(_DEF_SIG and 'style_inject' in _DEF_SIG.parameters)
_HAS_PULSE_SUBDIV = bool(_DEF_SIG and 'pulse_subdiv' in _DEF_SIG.parameters)

_HAS_VOCAL_ADAPTIVE = hasattr(sc, 'VocalAdaptive')
_HAS_SECTION_LFO = hasattr(sc, 'SectionLFO')
_HAS_MARKOV_PICK = hasattr(sc, 'markov_pick')
_HAS_APPEND_PHRASE = hasattr(sc, '_append_phrase')

# _dummy_pm may live in sc or in your test utilities; skip if unavailable.
if hasattr(sc, '_dummy_pm'):
    _dummy_pm = sc._dummy_pm  # type: ignore
else:
    _dummy_pm = None


# -----------------------
# Tests from codex branch
# -----------------------

@pytest.mark.skipif(not _HAS_APPEND_PHRASE, reason="_append_phrase not available")
def test_merge_reset_at_no_merge() -> None:
    inst = pretty_midi.Instrument(0)
    sc._append_phrase(inst, 60, 0.0, 1.0, 100, 0.1, 0.0, 0.0)
    sc._append_phrase(inst, 60, 1.05, 2.0, 100, -1.0, 0.0, 0.0)
    assert len(inst.notes) == 2


@pytest.mark.skipif(not (_HAS_SECTION_LFO and _dummy_pm), reason="SectionLFO or _dummy_pm missing")
def test_lfo_fill_velocity() -> None:
    pm = _dummy_pm()
    chords = [sc.ChordSpan(0, 2, 0, "maj")]
    mapping = {
        "phrase_note": 36,
        "phrase_velocity": 80,
        "phrase_length_beats": 1.0,
        "cycle_phrase_notes": [],
        "cycle_mode": "bar",
        "style_inject": {"period": 1, "note": 40},
        "lfo_apply": ["fill"],
    }
    lfo = sc.SectionLFO(4, vel_range=(0.5, 0.5), fill_range=(0.0, 0.0))
    out = sc.build_sparkle_midi(
        pm,
        chords,
        mapping,
        1.0,
        "bar",
        0.0,
        0,
        "flat",
        120,
        0.0,
        0.5,
        section_lfo=lfo,
        lfo_targets=("fill",),
    )
    vel = [n.velocity for n in out.instruments[1].notes if n.pitch == 40][0]
    assert vel == 40


@pytest.mark.skipif(not _HAS_VOCAL_ADAPTIVE, reason="VocalAdaptive not available")
def test_vocal_adapt_smoothing() -> None:
    va0 = sc.VocalAdaptive(5, 1, 2, [10, 0, 10])
    assert va0.phrase_for_bar(1) == 2
    va = sc.VocalAdaptive(5, 1, 2, [10, 0, 10], smooth_bars=2)
    assert va.phrase_for_bar(1) == 1


@pytest.mark.skipif(not _HAS_MARKOV, reason="schedule_phrase_keys markov not supported")
def test_markov_plan_and_fallback() -> None:
    markov = {"states": [24, 26], "T": [[0, 1], [0, 0]]}
    plan, *_ = sc.schedule_phrase_keys(3, None, None, None, markov=markov, rng=random.Random(0))
    assert plan == [24, 26, None]


@pytest.mark.skipif(not _HAS_MARKOV_PICK, reason="markov_pick not available")
def test_markov_degenerate_fallback() -> None:
    T = [[0, 0], [0, 0]]
    rng = random.Random(0)
    counts = {0: 0, 1: 0}
    for _ in range(10):
        counts[sc.markov_pick(T, 0, rng)] += 1
    assert counts[0] > 0 and counts[1] > 0


@pytest.mark.skipif(not (_HAS_BAR_QUALITIES), reason="bar_qualities not supported")
def test_harmony_weighting_changes_pool() -> None:
    plan, *_ = sc.schedule_phrase_keys(
        2,
        None,
        [{"start_bar": 0, "end_bar": 2, "phrase_pool": [36, 37]}],
        None,
        bar_qualities=["maj", "min"],
        rng=random.Random(0),
    )
    assert plan[0] == 36 and plan[1] == 37


@pytest.mark.skipif(not _HAS_SECTION_POOL_WEIGHTS, reason="section_pool_weights not supported")
def test_section_pool_weights_override() -> None:
    plan, *_ = sc.schedule_phrase_keys(
        1,
        None,
        [{"start_bar": 0, "end_bar": 1, "phrase_pool": [36, 37, 38], "tag": "verse"}],
        None,
        section_pool_weights={"verse": {38: 1.0}},
        rng=random.Random(0),
    )
    assert plan[0] == 38


@pytest.mark.skipif(not (_HAS_STYLE_INJECT and _HAS_PULSE_SUBDIV), reason="style_inject/pulse_subdiv not supported")
def test_smart_fill_prefers_section_end_and_respects_gap() -> None:
    _, fills, sources = sc.schedule_phrase_keys(
        5,
        None,
        [{"start_bar": 0, "end_bar": 4}],
        40,
        style_inject={"period": 5, "note": 40, "min_gap_beats": 4},
        pulse_subdiv=1.0,
        rng=random.Random(0),
    )
    assert set(fills.keys()) == {0, 3}
    assert sources[0] == "style" and sources[3] == "section"


@pytest.mark.skipif(not (_HAS_VOCAL_ADAPTIVE and _dummy_pm), reason="VocalAdaptive/_dummy_pm not available")
def test_vocal_ducking_reduces_velocity_and_prefers_muted() -> None:
    pm = _dummy_pm(2)
    chords = [sc.ChordSpan(0, 2, 0, "maj")]
    mapping = {
        "phrase_note": 36,
        "phrase_velocity": 100,
        "phrase_length_beats": 1.0,
        "cycle_phrase_notes": [],
        "cycle_mode": "bar",
    }
    va = sc.VocalAdaptive(1, 40, 36, [10])
    out = sc.build_sparkle_midi(
        pm,
        chords,
        mapping,
        1.0,
        "bar",
        0.0,
        0,
        "flat",
        120,
        0.0,
        0.5,
        vocal_adapt=va,
        vocal_ducking=0.5,
    )
    note = out.instruments[1].notes[0]
    assert note.pitch == 40 and note.velocity == 50


# ---------------------
# Tests from main branch
# ---------------------

@pytest.mark.skipif(not _dummy_pm, reason="_dummy_pm not available")
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
    stats = {"_legacy_bar_pulses_grid": True}
    out = sc.build_sparkle_midi(pm, chords, mapping, 0.5, 'bar', 0.0, 0,
                                'flat', 120, 0.0, 0.5,
                                section_profiles=profiles, sections=sections,
                                onset_list=[0,0,0,0], stats=stats)
    verse_notes = [n for n in out.instruments[1].notes if n.start < 4.0]
    chorus_notes = [n for n in out.instruments[1].notes if n.start >=4.0]
    assert any(n.pitch == 24 for n in verse_notes)
    assert any(n.pitch == 36 for n in chorus_notes)
    assert max(n.velocity for n in chorus_notes) > max(n.velocity for n in verse_notes)


@pytest.mark.skipif(not _dummy_pm, reason="_dummy_pm not available")
def test_style_layer_every() -> None:
    pm = _dummy_pm(8.0)
    chords = [sc.ChordSpan(i*2,(i+1)*2,0,'maj') for i in range(4)]
    mapping = {'phrase_note':24,'phrase_velocity':100,'phrase_length_beats':0.5,
               'cycle_phrase_notes':[], 'cycle_start_bar':0, 'cycle_mode':'bar'}
    stats = {"_legacy_bar_pulses_grid": True}
    out = sc.build_sparkle_midi(pm, chords, mapping, 0.5, 'bar', 0.0, 0,
                                'flat', 120, 0.0, 0.5, stats=stats)
    units = [(t, stats['downbeats'][i+1] if i+1 < len(stats['downbeats']) else pm.get_end_time())
             for i, t in enumerate(stats['downbeats'])]
    picker = sc.PoolPicker([(36,1)], rng=random.Random(0))
    sc.insert_style_layer(out, 'every', units, picker, every=2, length_beats=0.5)
    phrase_inst = [inst for inst in out.instruments if inst.name == sc.PHRASE_INST_NAME][0]
    starts = [round(n.start,2) for n in phrase_inst.notes if n.pitch==36]
    assert 0.0 in starts and 4.0 in starts


@pytest.mark.skipif(not _dummy_pm, reason="_dummy_pm not available")
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
    def travel(out_pm):
        notes = sorted(out_pm.instruments[0].notes, key=lambda n:n.start)
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


@pytest.mark.skipif(not _dummy_pm, reason="_dummy_pm not available")
def test_density_rules() -> None:
    pm = _dummy_pm(8.0)
    chords = [sc.ChordSpan(i*2,(i+1)*2,0,'maj') for i in range(4)]
    mapping = {'phrase_note':26,'phrase_velocity':100,'phrase_length_beats':0.5,
               'cycle_phrase_notes':[], 'cycle_start_bar':0, 'cycle_mode':'bar'}
    stats = {"_legacy_bar_pulses_grid": True}
    out = sc.build_sparkle_midi(pm, chords, mapping,0.5,'bar',0,0,'flat',120,0,0.5,
                                onset_list=[0,4,1,1], rest_list=[0.8,0.1,0.1,0.1], stats=stats)
    notes = [stats['bar_phrase_notes'].get(i) for i in range(4)]
    assert notes[0] == 24  # high rest -> open
    assert notes[1] == 36  # dense onsets -> high
    assert notes[2] == 26  # default


@pytest.mark.skipif(not _dummy_pm, reason="_dummy_pm not available")
def test_fill_cadence() -> None:
    pm = _dummy_pm(8.0)
    chords = [sc.ChordSpan(i*2,(i+1)*2,0,'maj') for i in range(4)]
    mapping = {'phrase_note':24,'phrase_velocity':100,'phrase_length_beats':0.5,
               'cycle_phrase_notes':[], 'cycle_start_bar':0, 'cycle_mode':'bar',
               'style_fill':34}
    stats = {"_legacy_bar_pulses_grid": True}
    out = sc.build_sparkle_midi(pm, chords, mapping,0.5,'bar',0,0,'flat',120,0,0.5, stats=stats)
    units = [(t, stats['downbeats'][i+1] if i+1 < len(stats['downbeats']) else pm.get_end_time())
             for i, t in enumerate(stats['downbeats'])]
    sections = [{'start_bar':0,'end_bar':2},{'start_bar':2,'end_bar':4}]
    cnt = sc.insert_style_fill(out, 'section_end', units, mapping,
                               sections=sections, min_gap_beats=0.5)
    assert cnt == 2
    phrase_inst = [inst for inst in out.instruments if inst.name == sc.PHRASE_INST_NAME][0]
    targets = {round(units[1][0], 2), round(units[3][0], 2)}
    fills = [
        n
        for n in phrase_inst.notes
        if round(n.start, 2) in targets and n.pitch != mapping['phrase_note']
    ]
    assert {round(n.start, 2) for n in fills} == targets
    assert all(n.pitch in {34, 35, 33} for n in fills)


@pytest.mark.skipif(not _dummy_pm, reason="_dummy_pm not available")
def test_swing_shapes() -> None:
    pm = _dummy_pm(4.0)
    chords = [sc.ChordSpan(0,4,0,'maj')]
    mapping = {'phrase_note':36,'phrase_velocity':100,'phrase_length_beats':0.5,
               'cycle_phrase_notes':[], 'cycle_start_bar':0, 'cycle_mode':'bar'}
    stats1 = {"_legacy_bar_pulses_grid": True}
    sc.build_sparkle_midi(pm, chords, mapping,0.5,'bar',0,0,'flat',120,0.5,0.5,
                          stats=stats1, swing_shape='offbeat')
    stats2 = {"_legacy_bar_pulses_grid": True}
    sc.build_sparkle_midi(pm, chords, mapping,0.5,'bar',0,0,'flat',120,0.5,0.5,
                          stats=stats2, swing_shape='even')
    pulses1 = [t for _, t in stats1['bar_pulses'][0][:3]]
    pulses2 = [t for _, t in stats2['bar_pulses'][0][:3]]
    intervals1 = [round(pulses1[i+1]-pulses1[i],3) for i in range(2)]
    intervals2 = [round(pulses2[i+1]-pulses2[i],3) for i in range(2)]
    assert intervals1 != intervals2


@pytest.mark.skipif(not _dummy_pm, reason="_dummy_pm not available")
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


@pytest.mark.skipif(not _dummy_pm, reason="_dummy_pm not available")
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


@pytest.mark.skipif(not _dummy_pm, reason="_dummy_pm not available")
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


@pytest.mark.skipif(not _dummy_pm, reason="_dummy_pm not available")
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





def test_merge_sections_cli_overrides_guide() -> None:
    merged = sc._merge_sections(
        [{"start_bar": 2, "tag": "B"}],
        ["a", "b", "c", "d"],
        10,
    )
    assert merged == [
        {"start_bar": 0, "end_bar": 1, "tag": "a", "source": "guide", "explicit_end": True},
        {"start_bar": 1, "end_bar": 2, "tag": "b", "source": "guide", "explicit_end": True},
        {"start_bar": 2, "end_bar": 3, "tag": "B", "source": "cli", "explicit_end": True},
        {"start_bar": 3, "end_bar": 10, "tag": "d", "source": "guide", "explicit_end": True},
    ]


def test_merge_sections_backfill_end() -> None:
    merged = sc._merge_sections(
        [{"start_bar": 5, "tag": "X"}],
        ["v", "p", "c"],
        8,
    )
    assert merged[-1]["start_bar"] == 5
    assert merged[-1]["end_bar"] == 8
    assert merged[-1]["tag"] == "X"


def test_merge_sections_overlap_autofix(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    cli = [{"start_bar": 0, "end_bar": 3, "tag": "intro"}, {"start_bar": 2, "tag": "bridge"}]
    guide = [
        {"start_bar": 0, "end_bar": 2, "tag": "A"},
        {"start_bar": 2, "end_bar": 4, "tag": "B"},
    ]
    merged = sc._merge_sections(cli, guide, 6)
    assert all(sec["start_bar"] < sec["end_bar"] for sec in merged)
    assert any("auto-fix" in rec.getMessage() for rec in caplog.records)


def test_insert_style_fill_section_end_ignores_tiny() -> None:
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0, name=sc.PHRASE_INST_NAME)
    pm.instruments.append(inst)
    units = [(0.0, 1.0), (1.0, 2.0)]
    count = sc.insert_style_fill(
        pm,
        "section_end",
        units,
        {"phrase_velocity": 96, "style_fill": 35},
        sections=[{"start_bar": 0, "end_bar": 0, "tag": "intro"}],
    )
    assert count == 0
    assert not inst.notes
def test_finalize_not_duplicated() -> None:
    pm = pretty_midi.PrettyMIDI()
    chord_inst = pretty_midi.Instrument(program=0, name=sc.CHORD_INST_NAME)
    phrase_inst = pretty_midi.Instrument(program=0, name=sc.PHRASE_INST_NAME)
    pm.instruments.extend([chord_inst, phrase_inst])
    stats = {
        "fill_bars": [],
        "sections": ["verse"],
        "bar_pulses": {0: []},
        "bar_pulse_grid": {0: []},
        "bar_triggers": {},
        "bar_phrase_notes": {},
        "bar_velocities": {},
        "downbeats": [0.0, 2.0],
        "bar_count": 1,
        "_legacy_bar_pulses_grid": True,
    }
    mapping = {"style_stop": 95}
    args = argparse.Namespace(
        auto_fill="off",
        fill_length_beats=0.25,
        fill_min_gap_beats=0.0,
        guide_rest_silence_th=None,
        debug_csv=None,
        bar_summary=None,
        report_json=None,
        section_default="verse",
    )
    sc.finalize_phrase_track(
        pm,
        args,
        stats,
        mapping,
        section_lfo=None,
        lfo_targets=(),
        downbeats=[0.0, 2.0],
        guide_units=[(0.0, 1.0), (1.0, 2.0)],
        guide_units_time=[(0.0, 1.0), (1.0, 2.0)],
        guide_notes={0: 60, 1: None},
        rest_ratios=[0.0, 1.0],
        onset_counts=[1, 0],
        chord_inst=chord_inst,
        phrase_inst=phrase_inst,
        beat_to_time=lambda b: b,
        time_to_beat=lambda t: t,
        pulse_subdiv_beats=1.0,
        phrase_vel=80,
        phrase_merge_gap=0.0,
        release_sec=0.0,
        min_phrase_len_sec=0.0,
        stop_min_gap_beats=0.0,
        stop_velocity=70,
        damp_dst=None,
        damp_cc_num=11,
        guide_cc=None,
        bpm=120.0,
        section_overrides=None,
        fill_map={0: (40, 0.5, 1.0)},
        rest_silence_send_stop=True,
        quantize_strength=1.0,
        write_markers=False,
        section_labels=["verse"],
        section_default="verse",
        chord_merge_gap=0.01,
        clone_meta_only=False,
        meta_src="test",
        chords=[sc.ChordSpan(0.0, 2.0, 0, "maj")],
    )
    assert stats["fill_count"] == 1
    stop_pitch = mapping["style_stop"]
    stop_notes = [n for n in phrase_inst.notes if n.pitch == stop_pitch]
    assert len(stop_notes) == 1
    fill_notes = [n for n in phrase_inst.notes if n.pitch == 40]
    assert len(fill_notes) == 1
    assert fill_notes[0].velocity == 80


def test_no_suppress_without_plan() -> None:
    plan, _, _ = sc.schedule_phrase_keys(
        3,
        [],
        sections=[{"start_bar": 0, "end_bar": 3, "phrase_pool": [36]}],
        fill_note=None,
        pulse_subdiv=1.0,
    )
    assert plan == [36, 36, 36]


def test_stats_triggers_vs_grid() -> None:
    if not hasattr(pretty_midi, "TimeSignature"):
        pytest.skip("pretty_midi stub lacks TimeSignature")
    pm = pretty_midi.PrettyMIDI()
    pm.time_signature_changes = [
        sc.pretty_midi.TimeSignature(6, 8, 0.0),
        sc.pretty_midi.TimeSignature(4, 4, 3.0),
    ]
    inst = pretty_midi.Instrument(program=0)
    inst.notes.append(pretty_midi.Note(velocity=80, pitch=60, start=0.0, end=7.0))
    pm.instruments.append(inst)
    chords = [
        sc.ChordSpan(0.0, 3.0, 0, "maj"),
        sc.ChordSpan(3.0, 7.0, 0, "maj"),
    ]
    mapping = {
        "phrase_note": 36,
        "phrase_velocity": 90,
        "phrase_length_beats": 1.0,
        "cycle_phrase_notes": [36],
        "cycle_mode": "bar",
    }
    stats: Dict[str, Any] = {"_legacy_bar_pulses_grid": True}
    out = sc.build_sparkle_midi(
        pm,
        chords,
        mapping,
        0.5,
        "bar",
        0.0,
        0,
        "flat",
        120.0,
        0.0,
        0.5,
        stats=stats,
    )
    assert len(stats["bar_pulse_grid"][0]) == len(stats["bar_pulses"][0]) == 6
    assert len(stats["bar_pulse_grid"][1]) == len(stats["bar_pulses"][1]) == 8
    phrase_inst = next(i for i in out.instruments if i.name == sc.PHRASE_INST_NAME)
    actual_counts = {0: 0, 1: 0}
    for note in phrase_inst.notes:
        bar_idx = 0 if note.start < stats["downbeats"][1] else 1
        actual_counts[bar_idx] += 1
    for bar_idx, triggers in stats["bar_triggers"].items():
        assert len(triggers) == actual_counts.get(bar_idx, 0)


def _pm_with_ts(num: int, den: int, length: float = 6.0):
    pm = pretty_midi.PrettyMIDI()
    pm.time_signature_changes.append(pretty_midi.TimeSignature(num, den, 0))
    inst = pretty_midi.Instrument(0)
    inst.notes.append(pretty_midi.Note(velocity=1, pitch=60, start=0.0, end=length))
    pm.instruments.append(inst)
    return pm


def _write_midi(tmp_path: Path) -> Path:
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0)
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=1.0))
    pm.instruments.append(inst)
    path = tmp_path / "in.mid"
    pm.write(str(path))
    return path


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
    chords = [sc.ChordSpan(0, 6, 0, "maj")]
    mapping = {
        "phrase_note": 36,
        "phrase_velocity": 100,
        "phrase_length_beats": 0.5,
        "cycle_phrase_notes": [],
        "cycle_start_bar": 0,
        "cycle_mode": "bar",
    }
    stats = {"_legacy_bar_pulses_grid": True}
    sc.build_sparkle_midi(
        pm, chords, mapping, 0.5, "bar", 0.0, 0, "flat", 120, 0.0, 0.5, stats=stats
    )
    assert len(stats["bar_pulses"][0]) == 12


def test_bar_pulses_12_8_swing_12() -> None:
    if not hasattr(pretty_midi, "TimeSignature"):
        pytest.skip("pretty_midi stub lacks TimeSignature")
    pm = _pm_with_ts(12, 8, 6.0)
    chords = [sc.ChordSpan(0, 6, 0, 'maj')]
    mapping = {'phrase_note': 36, 'phrase_velocity': 100, 'phrase_length_beats': 0.5,
               'cycle_phrase_notes': [], 'cycle_start_bar': 0, 'cycle_mode': 'bar'}
    stats = {"_legacy_bar_pulses_grid": True}
    sc.build_sparkle_midi(pm, chords, mapping, 0.5, 'bar', 0.0, 0, 'flat', 120, 0.0, 4/12,
                          stats=stats)
    assert len(stats['bar_pulses'][0]) == 12


def test_cycle_start_bar_negative() -> None:
    pm = _dummy_pm()
    chords = [sc.ChordSpan(0, 2, 0, "maj"), sc.ChordSpan(2, 4, 0, "maj")]
    mapping = {
        "phrase_note": 36,
        "phrase_velocity": 100,
        "phrase_length_beats": 0.25,
        "cycle_phrase_notes": [24, 26],
        "cycle_start_bar": -1,
        "cycle_mode": "bar",
    }
    out = sc.build_sparkle_midi(pm, chords, mapping, 1.0, "bar", 0.0, 0, "flat", 120, 0.0, 0.5)
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
    chords = [sc.ChordSpan(0, 2, 0, "maj")]
    mapping = {
        "phrase_note": 36,
        "phrase_velocity": 100,
        "phrase_length_beats": 0.25,
        "cycle_phrase_notes": [],
        "cycle_start_bar": 0,
        "cycle_mode": "bar",
        "top_note_max": 5,
        "strict": True,
    }
    with pytest.raises(SystemExit):
        sc.build_sparkle_midi(pm, chords, mapping, 1.0, "bar", 0.0, 0, "flat", 120, 0.0, 0.5)


def test_cycle_stride() -> None:
    pm = _dummy_pm(8.0)
    chords = [sc.ChordSpan(i * 2, (i + 1) * 2, 0, "maj") for i in range(4)]
    mapping = {
        "phrase_note": 36,
        "phrase_velocity": 100,
        "phrase_length_beats": 0.25,
        "cycle_phrase_notes": [24, 26],
        "cycle_start_bar": 0,
        "cycle_mode": "bar",
    }
    out = sc.build_sparkle_midi(
        pm, chords, mapping, 1.0, "bar", 0.0, 0, "flat", 120, 0.0, 0.5, cycle_stride=2
    )
    notes = out.instruments[1].notes
    assert all(n.pitch == 24 for n in notes if n.start < 4)
    assert all(n.pitch == 26 for n in notes if n.start >= 4)


def test_accent_profile() -> None:
    pm = _dummy_pm(4.0)
    chords = [sc.ChordSpan(0, 4, 0, "maj")]
    mapping = {
        "phrase_note": 36,
        "phrase_velocity": 100,
        "phrase_length_beats": 0.25,
        "cycle_phrase_notes": [],
        "cycle_start_bar": 0,
        "cycle_mode": "bar",
    }
    out = sc.build_sparkle_midi(
        pm, chords, mapping, 1.0, "bar", 0.0, 0, "flat", 120, 0.0, 0.5, accent=[1.0, 0.5]
    )
    vels = [n.velocity for n in out.instruments[1].notes[:4]]
    assert vels[0] == 100 and vels[1] == 50 and vels[2] == 100 and vels[3] == 50


def test_skip_phrase_in_rests() -> None:
    pm = _dummy_pm(4.0)
    chords = [sc.ChordSpan(0, 2, 0, "maj"), sc.ChordSpan(2, 4, 0, "rest")]
    mapping = {
        "phrase_note": 36,
        "phrase_velocity": 100,
        "phrase_length_beats": 0.25,
        "cycle_phrase_notes": [],
        "cycle_start_bar": 0,
        "cycle_mode": "bar",
        "silent_qualities": ["rest"],
    }
    out = sc.build_sparkle_midi(
        pm, chords, mapping, 1.0, "bar", 0.0, 0, "flat", 120, 0.0, 0.5, skip_phrase_in_rests=True
    )
    assert all(n.start < 2 for n in out.instruments[1].notes)


def test_phrase_chord_channels() -> None:
    pm = _dummy_pm()
    chords = [sc.ChordSpan(0, 2, 0, "maj")]
    mapping = {
        "phrase_note": 36,
        "phrase_velocity": 100,
        "phrase_length_beats": 0.25,
        "cycle_phrase_notes": [],
        "cycle_start_bar": 0,
        "cycle_mode": "bar",
    }
    out = sc.build_sparkle_midi(
        pm,
        chords,
        mapping,
        1.0,
        "bar",
        0.0,
        0,
        "flat",
        120,
        0.0,
        0.5,
        phrase_channel=1,
        chord_channel=2,
    )
    assert out.instruments[0].midi_channel == 2
    assert out.instruments[1].midi_channel == 1


@pytest.mark.skipif(
    pretty_midi.PrettyMIDI.__module__ == "tests._stubs",
    reason="pretty_midi stub lacks persistent channel handling",
)
def test_channel_roundtrip(tmp_path) -> None:
    pm = _dummy_pm()
    chords = [sc.ChordSpan(0, 2, 0, "maj")]
    mapping = {
        "phrase_note": 36,
        "phrase_velocity": 100,
        "phrase_length_beats": 0.25,
        "cycle_phrase_notes": [],
        "cycle_start_bar": 0,
        "cycle_mode": "bar",
    }
    out = sc.build_sparkle_midi(
        pm,
        chords,
        mapping,
        1.0,
        "bar",
        0.0,
        0,
        "flat",
        120,
        0.0,
        0.5,
        phrase_channel=1,
        chord_channel=2,
    )
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
        with mock.patch.object(
            sys,
            "argv",
            ["prog", "in.mid", "--out", str(out), "--dry-run", "--legacy-bar-pulses-grid"],
        ):
            with caplog.at_level(logging.INFO):
                sc.main()
        assert "bars=" in caplog.text
        assert "meter_map" in caplog.text
        assert not out.exists()
    finally:
        sc.pretty_midi.PrettyMIDI = orig


def test_pulse_grid_3_4_tempo_change() -> None:
    class Dummy:
        def __init__(self) -> None:
            self.instruments = []
            self.time_signature_changes = [
                types.SimpleNamespace(numerator=3, denominator=4, time=0.0)
            ]

        def get_beats(self):
            return [0.0, 0.5, 1.0, 1.5, 2.5, 3.5, 4.5]

        def get_downbeats(self):
            return []

        def get_end_time(self):
            return 4.5

        def get_tempo_changes(self):
            return [0.0], [120.0]

    pm = Dummy()
    chords = [sc.ChordSpan(0, 1.5, 0, "maj"), sc.ChordSpan(1.5, 4.5, 0, "maj")]
    mapping = {
        "phrase_note": 36,
        "phrase_velocity": 100,
        "phrase_length_beats": 0.25,
        "cycle_phrase_notes": [],
        "cycle_start_bar": 0,
        "cycle_mode": "bar",
    }
    out = sc.build_sparkle_midi(pm, chords, mapping, 1.0, "bar", 0.0, 0, "flat", 120, 0.0, 0.5)
    starts = [round(n.start, 3) for n in out.instruments[1].notes]
    assert starts == [0.0, 0.5, 1.0, 1.5, 2.5, 3.5]


def test_bar_width_6_8() -> None:
    class Dummy:
        def __init__(self) -> None:
            self.instruments = []
            self.time_signature_changes = [
                types.SimpleNamespace(numerator=6, denominator=8, time=0.0)
            ]

        def get_beats(self):
            return [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]

        def get_downbeats(self):
            return []

        def get_end_time(self):
            return 4.5

        def get_tempo_changes(self):
            return [0.0], [120.0]

    pm = Dummy()
    chords = [sc.ChordSpan(0, 4.5, 0, "maj")]
    mapping = {
        "phrase_note": 36,
        "phrase_velocity": 100,
        "phrase_length_beats": 0.25,
        "cycle_phrase_notes": [],
        "cycle_start_bar": 0,
        "cycle_mode": "bar",
    }
    out = sc.build_sparkle_midi(pm, chords, mapping, 1.0, "bar", 0.0, 0, "flat", 120, 0.0, 0.5)
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
    chords = [sc.ChordSpan(0, 1.0, 0, "maj")]
    mapping = {
        "phrase_note": 36,
        "phrase_velocity": 100,
        "phrase_length_beats": 0.25,
        "cycle_phrase_notes": [24, 26],
        "cycle_start_bar": 0,
        "cycle_mode": "bar",
    }
    with caplog.at_level(logging.INFO):
        sc.build_sparkle_midi(pm, chords, mapping, 1.0, "bar", 0.0, 0, "flat", 120, 0.0, 0.5)
    assert "cycle disabled; using fixed phrase_note=36" in caplog.text


def test_top_note_max() -> None:
    pm = _dummy_pm()
    chords = [sc.ChordSpan(0, 2, 0, "maj")]
    mapping = {
        "phrase_note": 36,
        "phrase_velocity": 100,
        "phrase_length_beats": 0.25,
        "cycle_phrase_notes": [],
        "cycle_start_bar": 0,
        "cycle_mode": "bar",
        "chord_octave": 5,
        "chord_input_range": {"lo": 60, "hi": 80},
        "voicing_mode": "stacked",
        "top_note_max": 70,
    }
    stats = {"_legacy_bar_pulses_grid": True}
    sc.build_sparkle_midi(
        pm, chords, mapping, 1.0, "bar", 0.0, 0, "flat", 120, 0.0, 0.5, stats=stats
    )
    triad = stats["triads"][0]
    assert max(triad) <= 70


def test_swing_timings() -> None:
    pm = _dummy_pm(3.0)
    chords = [sc.ChordSpan(0, 3, 0, "maj")]
    mapping = {
        "phrase_note": 36,
        "phrase_velocity": 100,
        "phrase_length_beats": 0.25,
        "cycle_phrase_notes": [],
        "cycle_start_bar": 0,
        "cycle_mode": "bar",
    }
    stats = {"_legacy_bar_pulses_grid": True}
    out = sc.build_sparkle_midi(
        pm, chords, mapping, 0.5, "bar", 0.0, 0, "flat", 120, 0.4, 0.5, stats=stats
    )
    pulses = stats["bar_pulses"][0]
    diff1 = pulses[1][0] - pulses[0][0]
    diff2 = pulses[2][0] - pulses[1][0]
    assert round(diff1, 2) == 0.7 and round(diff2, 2) == 0.3


def test_phrase_hold_chord_merges_pulses() -> None:
    pm = _dummy_pm()
    chords = [sc.ChordSpan(0, 2, 0, "maj"), sc.ChordSpan(2, 4, 0, "maj")]
    mapping = {
        "phrase_note": 36,
        "phrase_velocity": 100,
        "phrase_length_beats": 0.25,
        "cycle_phrase_notes": [],
        "cycle_start_bar": 0,
        "cycle_mode": "bar",
    }
    off = sc.build_sparkle_midi(pm, chords, mapping, 1.0, "bar", 0.0, 0, "flat", 120, 0.0, 0.5)
    mapping_hold = dict(mapping)
    mapping_hold.update({"phrase_hold": "chord", "phrase_merge_gap": 0.03})
    hold = sc.build_sparkle_midi(
        pm, chords, mapping_hold, 1.0, "bar", 0.0, 0, "flat", 120, 0.0, 0.5
    )
    assert len(hold.instruments[1].notes) < len(off.instruments[1].notes)


def test_phrase_release_and_minlen() -> None:
    pm = _dummy_pm()
    chords = [sc.ChordSpan(0, 2, 0, "maj")]
    mapping = {
        "phrase_note": 36,
        "phrase_velocity": 100,
        "phrase_length_beats": 0.25,
        "cycle_phrase_notes": [],
        "cycle_start_bar": 0,
        "cycle_mode": "bar",
        "phrase_release_ms": 50.0,
        "min_phrase_len_ms": 100.0,
    }
    out = sc.build_sparkle_midi(pm, chords, mapping, 1.0, "bar", 0.0, 0, "flat", 120, 0.0, 0.5)
    note = out.instruments[1].notes[0]
    length = note.end - note.start
    assert length >= 0.1 - sc.EPS and length < 0.125


def test_held_vel_mode_max() -> None:
    pm = _dummy_pm()
    chords = [sc.ChordSpan(0, 4, 0, "maj")]
    mapping_first = {
        "phrase_note": 36,
        "phrase_velocity": 100,
        "phrase_length_beats": 0.25,
        "cycle_phrase_notes": [],
        "cycle_start_bar": 0,
        "cycle_mode": "bar",
        "phrase_hold": "bar",
    }
    mapping_max = dict(mapping_first)
    mapping_max["held_vel_mode"] = "max"
    accent = [0.5, 1.0]
    out_first = sc.build_sparkle_midi(
        pm, chords, mapping_first, 1.0, "bar", 0.0, 0, "flat", 120, 0.0, 0.5, accent=accent
    )
    out_max = sc.build_sparkle_midi(
        pm, chords, mapping_max, 1.0, "bar", 0.0, 0, "flat", 120, 0.0, 0.5, accent=accent
    )
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
        val = kwargs.get("swing")
        if val is None and len(args) > 9:
            val = args[9]
        called["swing"] = val
        return pm

    sc.build_sparkle_midi = fake_build
    try:
        with mock.patch.object(
            sys,
            "argv",
            [
                "prog",
                "in.mid",
                "--out",
                str(out),
                "--dry-run",
                "--legacy-bar-pulses-grid",
                "--swing",
                "0.999",
            ],
        ):
            sc.main()
    finally:
        sc.build_sparkle_midi = orig_build
        sc.pretty_midi.PrettyMIDI = orig_pm
    assert called["swing"] <= 0.9


def test_cycle_token_parser() -> None:
    tokens = ["C2", "D#2", "rest", 36]
    assert [sc.parse_note_token(t) for t in tokens] == [36, 39, None, 36]


def test_merge_reset_at_bar() -> None:
    pm = _dummy_pm(4.0)
    chords = [sc.ChordSpan(0, 2, 0, "maj"), sc.ChordSpan(2, 4, 0, "maj")]
    mapping = {
        "phrase_note": 36,
        "phrase_velocity": 100,
        "phrase_length_beats": 1.0,
        "phrase_merge_gap": 0.1,
        "merge_reset_at": "bar",
        "cycle_phrase_notes": [],
        "cycle_start_bar": 0,
        "cycle_mode": "bar",
    }
    out = sc.build_sparkle_midi(
        pm, chords, mapping, 1.0, "bar", 0.0, 0, "flat", 120, 0.0, 1.0, merge_reset_at="bar"
    )
    notes = out.instruments[1].notes
    assert len(notes) == 2
    assert any(abs(n.start - 0.0) < 1e-6 for n in notes)
    assert any(abs(n.start - 2.0) < 1e-6 for n in notes)


# --- Guide-based features from codex/add-guide-midi-phrase-selection-and-damping ---

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


def test_insert_style_fill_with_label_sections() -> None:
    pm = _dummy_pm(8.0)
    chords = [sc.ChordSpan(i * 2, (i + 1) * 2, 0, 'maj') for i in range(4)]
    mapping = {'phrase_note': 36, 'phrase_velocity': 100, 'phrase_length_beats': 0.25,
               'cycle_phrase_notes': [], 'cycle_start_bar': 0, 'cycle_mode': 'bar'}
    stats: Dict[str, Any] = {"_legacy_bar_pulses_grid": True}
    out = sc.build_sparkle_midi(pm, chords, mapping, 1.0, 'bar', 0.0, 0, 'flat', 120, 0.0, 0.5,
                                stats=stats)
    downbeats = stats.get('downbeats') or []
    units = [(downbeats[i], downbeats[i + 1]) for i in range(len(downbeats) - 1)]
    cnt = sc.insert_style_fill(out, 'section_end', units, mapping,
                               sections=['intro', 'verse', 'chorus'], bpm=120.0)
    assert cnt >= 1


def test_insert_style_fill_avoid_overlap() -> None:
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0, name=sc.PHRASE_INST_NAME)
    for bar in range(4):
        inst.notes.append(pretty_midi.Note(velocity=90, pitch=34, start=bar, end=bar + 0.3))
    pm.instruments.append(inst)
    units = [(float(i), float(i + 1)) for i in range(4)]
    sections = [{"start_bar": i, "end_bar": i + 1, "tag": "sec"} for i in range(4)]
    mapping = {
        'phrase_note': 36,
        'phrase_velocity': 100,
        'phrase_length_beats': 0.25,
        'cycle_phrase_notes': [],
        'style_fill': 34,
    }
    cnt = sc.insert_style_fill(
        pm,
        'section_end',
        units,
        mapping,
        sections=sections,
        bpm=120.0,
        min_gap_beats=0.25,
        bar_count=4,
        section_default='sec',
    )
    assert cnt == 4
    new_notes = inst.notes[4:]
    assert new_notes
    assert all(n.pitch != 34 for n in new_notes)
    assert len(new_notes) == 4
    assert {n.pitch for n in new_notes} == {35}


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
    stats = {"_legacy_bar_pulses_grid": True}
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
    assert cnt == 1
    assert len(inst.notes) == 3
    assert inst.notes[-1].pitch != 36


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
    stats = {"_legacy_bar_pulses_grid": True}
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
    stats = {"_legacy_bar_pulses_grid": True}
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


# --- Scheduler-based features from main ---

def test_scheduler_fill_once_on_section_end() -> None:
    pm = _dummy_pm(8.0)
    chords = [sc.ChordSpan(0, 8, 0, "maj")]
    mapping = {
        "phrase_note": 36,
        "phrase_velocity": 100,
        "phrase_length_beats": 0.25,
        "cycle_phrase_notes": [],
        "cycle_mode": "bar",
        "phrase_hold": "bar",
        "sections": [
            {"start_bar": 0, "end_bar": 2, "tag": "Verse"},
            {"start_bar": 2, "end_bar": 4, "tag": "Chorus"},
        ],
        "style_fill": 35,
    }
    out = sc.build_sparkle_midi(pm, chords, mapping, 1.0, "bar", 0.0, 0, "flat", 120, 0.0, 0.5)
    phrase_notes = out.instruments[1].notes
    fills = [n for n in phrase_notes if n.pitch == 35]
    assert len(fills) == 1


def test_scheduler_intensity_ramp() -> None:
    plan, fills, _ = sc.schedule_phrase_keys(4, None, None, None)
    assert plan == [None, None, None, None]
    assert fills == {}


def test_scheduler_respects_hold_only_changes_on_switch() -> None:
    pm = _dummy_pm(6.0)
    chords = [sc.ChordSpan(0, 6, 0, "maj")]
    mapping = {
        "phrase_note": 36,
        "phrase_velocity": 100,
        "phrase_length_beats": 0.25,
        "cycle_phrase_notes": [36, 36, 38],
        "cycle_mode": "bar",
        "phrase_hold": "bar",
    }
    out = sc.build_sparkle_midi(pm, chords, mapping, 1.0, "bar", 0.0, 0, "flat", 120, 0.0, 0.5)
    pitches = [n.pitch for n in out.instruments[1].notes]
    assert pitches == [36, 38]


def test_section_lfo_velocity_and_fill_arc() -> None:
    pm = _dummy_pm(8.0)
    chords = [sc.ChordSpan(i * 2, (i + 1) * 2, 0, "maj") for i in range(4)]
    mapping = {
        "phrase_note": 36,
        "phrase_velocity": 100,
        "phrase_length_beats": 0.5,
        "cycle_phrase_notes": [],
        "cycle_start_bar": 0,
        "cycle_mode": "bar",
        "style_fill": 35,
    }
    lfo = sc.SectionLFO(
        bars_period=4, vel_range=(0.5, 1.0), fill_range=(0.0, 1.0), shape="triangle"
    )
    stats = {"_legacy_bar_pulses_grid": True}
    sc.build_sparkle_midi(
        pm, chords, mapping, 0.5, "bar", 0.0, 0, "flat", 120, 0.0, 0.5, stats=stats, section_lfo=lfo
    )
    assert stats["bar_velocities"][0][0] < stats["bar_velocities"][3][0]
    assert stats["fill_count"] == 1


def test_stable_chord_guard_alternate() -> None:
    pm = _dummy_pm(8.0)
    chords = [sc.ChordSpan(0, 8, 0, "maj")]
    mapping = {
        "phrase_note": 36,
        "phrase_velocity": 100,
        "phrase_length_beats": 0.5,
        "cycle_phrase_notes": [],
        "cycle_start_bar": 0,
        "cycle_mode": "chord",
        "phrase_merge_gap": 0.0,
        "phrase_release_ms": 10,
    }
    stats_no = {}
    sc.build_sparkle_midi(
        pm, chords, mapping, 0.5, "chord", 0.0, 0, "flat", 120, 0.0, 0.5, stats=stats_no
    )
    guard = sc.StableChordGuard(min_hold_beats=4, strategy="alternate")
    stats_guard = {}
    sc.build_sparkle_midi(
        pm,
        chords,
        mapping,
        0.5,
        "chord",
        0.0,
        0,
        "flat",
        120,
        0.0,
        0.5,
        stable_guard=guard,
        stats=stats_guard,
    )
    assert stats_guard["pulse_count"] < stats_no["pulse_count"]


def test_vocal_adapt_density_switch() -> None:
    pm = _dummy_pm(4.0)
    chords = [sc.ChordSpan(0, 2, 0, "maj"), sc.ChordSpan(2, 4, 0, "maj")]
    mapping = {
        "phrase_note": 36,
        "phrase_velocity": 100,
        "phrase_length_beats": 0.25,
        "cycle_phrase_notes": [41],
        "cycle_start_bar": 0,
        "cycle_mode": "bar",
    }
    va = sc.VocalAdaptive(dense_onset=2, dense_phrase=40, sparse_phrase=41, onsets=[3, 0])
    stats = {"_legacy_bar_pulses_grid": True}
    sc.build_sparkle_midi(
        pm, chords, mapping, 1.0, "bar", 0.0, 0, "flat", 120, 0.0, 0.5, stats=stats, vocal_adapt=va
    )
    assert stats["bar_phrase_notes"][0] == 40



def test_no_suppress_without_plan() -> None:
    plan, _, _ = sc.schedule_phrase_keys(
        3,
        [],
        sections=[{"start_bar": 0, "end_bar": 3, "phrase_pool": [36]}],
        fill_note=None,
        pulse_subdiv=1.0,
    )
    assert plan == [36, 36, 36]


def test_stats_triggers_vs_grid() -> None:
    if not hasattr(pretty_midi, "TimeSignature"):
        pytest.skip("pretty_midi stub lacks TimeSignature")
    pm = pretty_midi.PrettyMIDI()
    pm.time_signature_changes = [
        sc.pretty_midi.TimeSignature(6, 8, 0.0),
        sc.pretty_midi.TimeSignature(4, 4, 3.0),
    ]
    inst = pretty_midi.Instrument(program=0)
    inst.notes.append(pretty_midi.Note(velocity=80, pitch=60, start=0.0, end=7.0))
    pm.instruments.append(inst)
    chords = [
        sc.ChordSpan(0.0, 3.0, 0, "maj"),
        sc.ChordSpan(3.0, 7.0, 0, "maj"),
    ]
    mapping = {
        "phrase_note": 36,
        "phrase_velocity": 90,
        "phrase_length_beats": 1.0,
        "cycle_phrase_notes": [36],
        "cycle_mode": "bar",
    }
    stats: dict[str, Any] = {}
    out = sc.build_sparkle_midi(
        pm,
        chords,
        mapping,
        0.5,
        "bar",
        0.0,
        0,
        "flat",
        120.0,
        0.0,
        0.5,
        stats=stats,
    )
    assert len(stats["bar_pulse_grid"][0]) == len(stats["bar_pulses"][0]) == 6
    assert len(stats["bar_pulse_grid"][1]) == len(stats["bar_pulses"][1]) == 8
    phrase_inst = next(i for i in out.instruments if i.name == sc.PHRASE_INST_NAME)
    actual_counts = {0: 0, 1: 0}
    for note in phrase_inst.notes:
        bar_idx = 0 if note.start < stats["downbeats"][1] else 1
        actual_counts[bar_idx] += 1
    for bar_idx, triggers in stats["bar_triggers"].items():
        assert len(triggers) == actual_counts.get(bar_idx, 0)


def _pm_with_ts_seq(seq):
    pm = pretty_midi.PrettyMIDI()
    t = 0.0
    inst = pretty_midi.Instrument(0)
    for num, den, dur in seq:
        pm.time_signature_changes.append(pretty_midi.TimeSignature(num, den, t))
        t += dur
    inst.notes.append(pretty_midi.Note(velocity=1, pitch=60, start=0.0, end=t))
    pm.instruments.append(inst)
    return pm


def test_meter_change_5_4_and_7_8() -> None:
    if not hasattr(pretty_midi, "TimeSignature"):
        pytest.skip("pretty_midi stub lacks TimeSignature")
    seq = [(5, 4, 2.5), (7, 8, 1.75)]
    pm = _pm_with_ts_seq(seq)
    chords = [sc.ChordSpan(0, 2.5, 0, "maj"), sc.ChordSpan(2.5, 4.25, 0, "maj")]
    mapping = {
        "phrase_note": 36,
        "phrase_velocity": 100,
        "phrase_length_beats": 0.5,
        "cycle_phrase_notes": [],
        "cycle_start_bar": 0,
        "cycle_mode": "bar",
    }
    stats = {"_legacy_bar_pulses_grid": True}
    sc.build_sparkle_midi(
        pm, chords, mapping, 0.5, "bar", 0.0, 0, "flat", 120, 0.0, 0.5, stats=stats
    )
    assert len(stats["bar_pulses"][0]) == 10
    assert len(stats["bar_pulses"][1]) == 7


def test_meter_change_6_8_to_4_4() -> None:
    if not hasattr(pretty_midi, "TimeSignature"):
        pytest.skip("pretty_midi stub lacks TimeSignature")
    seq = [(6, 8, 1.5), (4, 4, 2.0)]
    pm = _pm_with_ts_seq(seq)
    chords = [sc.ChordSpan(0, 1.5, 0, "maj"), sc.ChordSpan(1.5, 3.5, 0, "maj")]
    mapping = {
        "phrase_note": 36,
        "phrase_velocity": 100,
        "phrase_length_beats": 0.5,
        "cycle_phrase_notes": [],
        "cycle_start_bar": 0,
        "cycle_mode": "bar",
    }
    stats = {"_legacy_bar_pulses_grid": True}
    sc.build_sparkle_midi(
        pm, chords, mapping, 0.5, "bar", 0.0, 0, "flat", 120, 0.0, 0.5, stats=stats
    )
    assert len(stats["bar_pulses"][0]) == 6
    assert len(stats["bar_pulses"][1]) == 8


def test_section_profile_partial_override() -> None:
    plan, _, _ = sc.schedule_phrase_keys(
        4, [24, 26], [{"start_bar": 1, "end_bar": 3, "pool": [30]}], None
    )
    assert plan == [24, 30, 30, 26]


def test_markov_degenerate_fallback() -> None:
    T = [[0, 0], [0, 0]]
    counts = {0: 0, 1: 0}
    rng = random.Random(0)
    for _ in range(10):
        counts[sc.markov_pick(T, 0, rng)] += 1
    assert counts[0] > 0 and counts[1] > 0


def test_guard_hold_beats_variable_intervals() -> None:
    class TempoPM:
        def __init__(self):
            inst = pretty_midi.Instrument(0)
            inst.notes.append(pretty_midi.Note(velocity=1, pitch=60, start=0.0, end=3.5))
            inst.is_drum = False
            self.instruments = [inst]
            self.time_signature_changes = []

        def get_beats(self):
            return [0.0, 0.5, 1.0, 1.5, 2.5, 3.5]

        def get_downbeats(self):
            return [0.0, 2.5]

        def get_end_time(self):
            return 3.5

        def get_tempo_changes(self):
            return [0.0], [120.0]

    pm = TempoPM()
    chords = [sc.ChordSpan(0, 3.5, 0, "maj")]
    mapping = {
        "phrase_note": 36,
        "phrase_velocity": 100,
        "phrase_length_beats": 0.5,
        "cycle_phrase_notes": [],
        "cycle_start_bar": 0,
        "cycle_mode": "bar",
    }
    guard = sc.StableChordGuard(min_hold_beats=2, strategy="alternate")
    stats = {}
    sc.build_sparkle_midi(
        pm,
        chords,
        mapping,
        0.5,
        "bar",
        0.0,
        0,
        "flat",
        120,
        0.0,
        0.5,
        stats=stats,
        stable_guard=guard,
    )
    assert stats["guard_hold_beats"][0] == pytest.approx(4.0, abs=1e-6)


def test_lfo_apply_to_chord_velocity() -> None:
    pm = _dummy_pm(8.0)
    chords = [sc.ChordSpan(i * 2, (i + 1) * 2, 0, "maj") for i in range(4)]
    mapping = {
        "phrase_note": 36,
        "phrase_velocity": 100,
        "phrase_length_beats": 0.5,
        "cycle_phrase_notes": [],
        "cycle_start_bar": 0,
        "cycle_mode": "bar",
    }
    lfo = sc.SectionLFO(bars_period=4, vel_range=(0.5, 1.0))
    out1 = sc.build_sparkle_midi(
        pm,
        chords,
        mapping,
        0.5,
        "bar",
        0.0,
        0,
        "flat",
        120,
        0.0,
        0.5,
        section_lfo=lfo,
        lfo_targets=("phrase",),
    )
    out2 = sc.build_sparkle_midi(
        pm,
        chords,
        mapping,
        0.5,
        "bar",
        0.0,
        0,
        "flat",
        120,
        0.0,
        0.5,
        section_lfo=lfo,
        lfo_targets=("phrase", "chord"),
    )
    assert out2.instruments[0].notes[0].velocity < out1.instruments[0].notes[0].velocity


def test_vocal_guide_from_midi_density_switch(tmp_path) -> None:
    pm_v = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0)
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=0.2))
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=62, start=0.3, end=0.5))
    pm_v.instruments.append(inst)
    vocal_path = tmp_path / "v.mid"
    pm_v.write(str(vocal_path))
    counts = sc.vocal_onsets_from_midi(str(vocal_path))
    counts = [2, 0]
    va = sc.VocalAdaptive(dense_onset=1, dense_phrase=40, sparse_phrase=41, onsets=counts)
    pm = _dummy_pm(2.0)
    chords = [sc.ChordSpan(0, 1.0, 0, "maj"), sc.ChordSpan(1.0, 2.0, 0, "maj")]
    mapping = {
        "phrase_note": 36,
        "phrase_velocity": 100,
        "phrase_length_beats": 0.5,
        "cycle_phrase_notes": [],
        "cycle_start_bar": 0,
        "cycle_mode": "bar",
    }
    stats = {}
    sc.build_sparkle_midi(
        pm, chords, mapping, 0.5, "bar", 0.0, 0, "flat", 120, 0.0, 0.5, stats=stats, vocal_adapt=va
    )
    assert stats["bar_phrase_notes"][0] == 40


def test_periodic_style_injection_and_conflict_policy() -> None:
    plan, fills, src = sc.schedule_phrase_keys(
        4, None, [{"end_bar": 2}], 35, style_inject={"period": 2, "note": 30}
    )
    assert fills[0][0] == 30
    assert fills[1][0] == 35
    assert fills[2][0] == 30


def test_stats_traceability_keys_present() -> None:
    pm = _dummy_pm(4.0)
    chords = [sc.ChordSpan(0, 4.0, 0, "maj")]
    mapping = {
        "phrase_note": 36,
        "phrase_velocity": 100,
        "phrase_length_beats": 0.5,
        "cycle_phrase_notes": [],
        "cycle_start_bar": 0,
        "cycle_mode": "bar",
    }
    lfo = sc.SectionLFO(bars_period=2, vel_range=(0.5, 1.0))
    guard = sc.StableChordGuard(min_hold_beats=1, strategy="alternate")
    stats = {}
    sc.build_sparkle_midi(
        pm,
        chords,
        mapping,
        0.5,
        "bar",
        0.0,
        0,
        "flat",
        120,
        0.0,
        0.5,
        stats=stats,
        section_lfo=lfo,
        stable_guard=guard,
    )
    assert stats["bar_reason"] and stats["lfo_pos"] and stats["guard_hold_beats"]


def test_config_validation() -> None:
    with pytest.raises(SystemExit):
        sc.validate_section_lfo_cfg({"period": 0})
    with pytest.raises(SystemExit):
        sc.validate_section_lfo_cfg({"period": 2, "vel": [1.0]})
    with pytest.raises(SystemExit):
        sc.validate_section_lfo_cfg({"period": 2, "fill": [0, 1, 2]})
    with pytest.raises(SystemExit):
        sc.validate_style_inject_cfg({"period": 0, "note": 30})
    with pytest.raises(SystemExit):
        sc.validate_vocal_adapt_cfg({"dense_onset": 1, "dense_ratio": 2})
    with pytest.raises(SystemExit):
        sc.validate_stable_guard_cfg({"min_hold_beats": -1})


def test_lfo_shapes_linear_sine_triangle() -> None:
    lin = sc.SectionLFO(bars_period=4, vel_range=(0.0, 1.0), shape="linear")
    sine = sc.SectionLFO(bars_period=4, vel_range=(0.0, 1.0), shape="sine")
    tri = sc.SectionLFO(bars_period=4, vel_range=(0.0, 1.0), shape="triangle")
    vals_lin = [lin.vel_scale(i) for i in range(4)]
    vals_sine = [sine.vel_scale(i) for i in range(4)]
    vals_tri = [tri.vel_scale(i) for i in range(4)]
    assert vals_lin[1] < vals_lin[2] < vals_lin[3]
    assert vals_sine[1] > vals_lin[1]
    assert vals_tri[2] == pytest.approx(1.0)


def test_fill_sources_and_policy() -> None:
    lfo = sc.SectionLFO(bars_period=1, fill_range=(1.0, 1.0))
    _, _, src = sc.schedule_phrase_keys(
        2,
        None,
        [{"end_bar": 2}],
        35,
        lfo=lfo,
        style_inject={"period": 1, "note": 30},
        fill_policy="lfo",
    )
    assert src[0] == "lfo" and src[1] == "lfo"
    _, _, src2 = sc.schedule_phrase_keys(
        2,
        None,
        [{"end_bar": 2}],
        35,
        lfo=lfo,
        style_inject={"period": 1, "note": 30},
        fill_policy="style",
    )
    assert src2[0] == "style" and src2[1] == "style"


def test_vocal_density_by_ratio(tmp_path) -> None:
    on, rat = [1], [0.8]
    va = sc.VocalAdaptive(
        dense_onset=5, dense_phrase=40, sparse_phrase=41, onsets=on, ratios=rat, dense_ratio=0.5
    )
    assert va.phrase_for_bar(0) == 40


def test_style_inject_duration_and_velocity() -> None:
    pm = _dummy_pm(4.0)
    chords = [sc.ChordSpan(0, 4.0, 0, "maj")]
    mapping = {
        "phrase_note": 36,
        "phrase_velocity": 80,
        "phrase_length_beats": 0.5,
        "cycle_phrase_notes": [],
        "cycle_start_bar": 0,
        "cycle_mode": "bar",
        "style_inject": {"period": 2, "note": 30, "duration_beats": 2, "vel_scale": 1.5},
    }
    out = sc.build_sparkle_midi(
        pm, chords, mapping, 0.5, "bar", 0.0, 0, "flat", 120, 0.0, 0.5, fill_policy="style"
    )
    fills = [n for n in out.instruments[1].notes if n.pitch == 30]
    assert fills and (fills[0].end - fills[0].start) == pytest.approx(1.0, abs=1e-3)
    assert fills[0].velocity > 80


def test_section_profiles_basic(tmp_path) -> None:
    yaml_text = (
        '{"sections":[{"tag":"verse","bars":[0,2],"phrase_pool":[36,"rest"],"pool_weights":[1,0]}]}'
    )
    p = tmp_path / "sections.yaml"
    p.write_text(yaml_text)
    secs = sc.read_section_profiles(p)
    stats = {}
    plan, fills, _ = sc.schedule_phrase_keys(4, None, secs, None, rng=random.Random(0), stats=stats)
    assert plan[0] == 36 and plan[1] == 36
    assert stats["bar_reason"][0]["source"] == "section"


def test_section_fill_cadence(tmp_path) -> None:
    yaml_text = '{"sections":[{"tag":"verse","bars":[0,4],"phrase_pool":[36],"fill_cadence":2}]}'
    p = tmp_path / "sections.yaml"
    p.write_text(yaml_text)
    secs = sc.read_section_profiles(p)
    plan, fills, src = sc.schedule_phrase_keys(4, None, secs, 35, rng=random.Random(0))
    assert sorted(fills.keys()) == [1, 3]


def test_damping_vocal_envelope() -> None:
    pm = pretty_midi.PrettyMIDI()
    sc.emit_damping(
        pm,
        "vocal",
        downbeats=[0.0, 1.0, 2.0],
        vocal_ratios=[0.0, 1.0, 0.0],
        cc=11,
        min_beats=0.0,
        deadband=0.0,
        clip=(0, 127),
    )
    inst = pm.instruments[-1]
    vals = [(cc.time, cc.value) for cc in inst.control_changes]
    assert vals == [(0.0, 0), (1.0, 127), (2.0, 0)]


def test_seed_reproducibility() -> None:
    secs = [
        {"tag": "v", "start_bar": 0, "end_bar": 4, "phrase_pool": [36, 38], "pool_weights": [1, 1]}
    ]
    rng1 = random.Random(42)
    plan1, fills1, _ = sc.schedule_phrase_keys(4, None, secs, 35, rng=rng1)
    rng2 = random.Random(42)
    plan2, fills2, _ = sc.schedule_phrase_keys(4, None, secs, 35, rng=rng2)
    assert plan1 == plan2 and fills1 == fills2


def test_yaml_errors_are_helpful(tmp_path) -> None:
    bad = '{"sections":[{"tag":"verse","bars":[0],"phrase_pool":[36]}]}'
    p = tmp_path / "bad.yaml"
    p.write_text(bad)
    with pytest.raises(ValueError) as e:
        sc.read_section_profiles(p)
    assert "sections[0].bars" in str(e.value)


def test_phrase_note_aliases() -> None:
    sc.NOTE_ALIASES = {"open_1_8": 24}
    tokens = ["C1", "open_1_8", "rest"]
    assert [sc.parse_note_token(t) for t in tokens] == [24, 24, None]


def test_section_density_presets() -> None:
    pm = _dummy_pm(4.0)
    chords = [sc.ChordSpan(0, 2.0, 0, "maj"), sc.ChordSpan(2.0, 4.0, 0, "maj")]
    mapping = {
        "phrase_note": 24,
        "phrase_velocity": 80,
        "phrase_length_beats": 0.25,
        "sections": [
            {"start_bar": 0, "end_bar": 1, "density": "low"},
            {"start_bar": 1, "end_bar": 2, "density": "high"},
        ],
    }
    stats = {}
    sc.build_sparkle_midi(
        pm, chords, mapping, 0.5, "bar", 0.0, 0, "flat", 120, 0.0, 0.5, stats=stats
    )
    triggers = stats.get("bar_triggers", {})
    p0 = len(triggers.get(0, []))
    p1 = len(triggers.get(1, []))
    v0 = sum(stats["bar_velocities"][0]) / len(stats["bar_velocities"][0])
    v1 = sum(stats["bar_velocities"][1]) / len(stats["bar_velocities"][1])
    assert p0 < p1 and v0 < v1


def test_pool_by_quality() -> None:
    sections = [{"start_bar": 0, "end_bar": 2, "pool_by_quality": {"maj": [36], "min": [37]}}]
    plan, _, _ = sc.schedule_phrase_keys(2, None, sections, None, bar_qualities=["maj", "min"])
    assert plan == [36, 37]


def test_damping_cli_options() -> None:
    pm = pretty_midi.PrettyMIDI()
    sc.emit_damping(
        pm,
        "vocal",
        downbeats=[0.0, 1.0, 2.0],
        vocal_ratios=[0.0, 1.0, 0.0],
        cc=11,
        channel=1,
        smooth=1,
        deadband=0.0,
        min_beats=0.0,
        clip=(0, 100),
    )
    inst = pm.instruments[-1]
    assert inst.midi_channel == 1 and inst.control_changes[1].value <= 100


def test_accent_stretching() -> None:
    pm = _dummy_pm(2.0)
    chords = [sc.ChordSpan(0, 2.0, 0, "maj")]
    mapping = {
        "phrase_note": 24,
        "phrase_velocity": 80,
        "phrase_length_beats": 0.5,
        "accent": [1.0, 0.5],
    }
    stats = {}
    sc.build_sparkle_midi(
        pm,
        chords,
        mapping,
        0.5,
        "bar",
        0.0,
        0,
        "flat",
        120,
        0.0,
        0.5,
        stats=stats,
        accent=[1.0, 0.5],
    )
    vels = stats["bar_velocities"][0]
    assert len(set(vels)) > 1


def test_write_reports(tmp_path) -> None:
    stats = {"bar_count": 2, "fill_count": 1}
    jp = tmp_path / "r.json"
    mp = tmp_path / "r.md"
    sc.write_reports(stats, jp, mp)
    assert "bar_count" in jp.read_text()
    assert "Sparkle Report" in mp.read_text()


def test_section_preset_and_override() -> None:
    sc.NOTE_ALIASES = {"open_1_8": 24, "open_1_16": 26, "muted_1_8": 25}
    sc.NOTE_ALIAS_INV = {24: "open_1_8", 26: "open_1_16", 25: "muted_1_8"}
    mapping = {
        "sections": [
            {"tag": "verse", "start_bar": 0, "end_bar": 2},
            {"tag": "chorus", "start_bar": 2, "end_bar": 4, "phrase_pool": [25]},
        ]
    }
    sc.apply_section_preset(mapping, "acoustic_ballad")
    plan, _, _ = sc.schedule_phrase_keys(4, None, mapping["sections"], None)
    assert plan[:2] == [24, 24]
    assert plan[2:4] == [25, 25]


def test_vocal_guide_style_fill() -> None:
    pm = _dummy_pm(8.0)
    chords = [sc.ChordSpan(0, 8.0, 0, "maj")]
    mapping = {"phrase_note": 36, "phrase_velocity": 80, "phrase_length_beats": 1.0}
    onsets = [0, 0, 5, 0]
    out = sc.build_sparkle_midi(
        pm,
        chords,
        mapping,
        1.0,
        "bar",
        0.0,
        0,
        "flat",
        120,
        0.0,
        0.5,
        guide_onsets=onsets,
        guide_onset_th=3,
        guide_style_note=40,
    )
    fills = [n for n in out.instruments[1].notes if n.pitch == 40]
    assert len(fills) == 1 and 2.0 <= fills[0].start < 4.0


def test_swing_hold_merge() -> None:
    pm = _dummy_pm(8.0)
    chords = [sc.ChordSpan(0, 4.0, 0, "maj"), sc.ChordSpan(4.0, 8.0, 0, "maj")]
    mapping = {
        "phrase_note": 36,
        "phrase_velocity": 80,
        "phrase_length_beats": 0.5,
        "phrase_hold": "chord",
        "phrase_merge_gap": 0.1,
        "cycle_phrase_notes": [],
        "cycle_mode": "bar",
    }
    out = sc.build_sparkle_midi(pm, chords, mapping, 0.5, "bar", 0.0, 0, "flat", 120, 0.2, 0.5)
    notes = out.instruments[1].notes
    assert len(notes) == 2


def test_debug_md_output(tmp_path) -> None:
    pm = _dummy_pm(4.0)
    chords = [sc.ChordSpan(0, 4.0, 0, "maj")]
    mapping = {"phrase_note": 36, "phrase_velocity": 80, "phrase_length_beats": 1.0}
    stats = {}
    sc.build_sparkle_midi(
        pm, chords, mapping, 1.0, "bar", 0.0, 0, "flat", 120, 0.0, 0.5, stats=stats
    )
    p = tmp_path / "dbg.md"
    sc.write_debug_md(stats, p)
    text = p.read_text().splitlines()
    assert text[0].startswith("|bar|") and len(text) >= 3


def test_section_profile_alias_warning(tmp_path, caplog) -> None:
    pytest.importorskip("yaml")
    y = "sections:\n- {tag: verse, bars: [0,1], phrase_pool: [unknown_alias]}\n"
    p = tmp_path / "s.yaml"
    p.write_text(y)
    with caplog.at_level(logging.WARNING):
        secs = sc.read_section_profiles(p, {"open_1_8": 24})
    assert secs[0]["phrase_pool"][0] is None
    assert any("unknown note alias" in r.message for r in caplog.records)


def test_seed_cli_repro(tmp_path) -> None:
    inp = _write_midi(tmp_path)
    out = tmp_path / "o.mid"
    j1 = tmp_path / "r1.json"
    j2 = tmp_path / "r2.json"
    argv = [
        "sc",
        str(inp),
        "--out",
        str(out),
        "--pulse",
        "1/8",
        "--cycle-phrase-notes",
        "24,26",
        "--humanize-vel",
        "5",
        "--seed",
        "7",
        "--report-json",
        str(j1),
        "--dry-run",
        "--legacy-bar-pulses-grid",
    ]

    def fake_pm(*a, **k):
        pm = _dummy_pm(2.0)
        if not a:
            pm.instruments = []
        created.append(pm)
        return pm

    created = []
    with mock.patch.object(sc.pretty_midi, "PrettyMIDI", side_effect=fake_pm):
        with mock.patch.object(sys, "argv", argv):
            sc.main()
    argv[argv.index("--report-json") + 1] = str(j2)
    created = []
    with mock.patch.object(sc.pretty_midi, "PrettyMIDI", side_effect=fake_pm):
        with mock.patch.object(sys, "argv", argv):
            sc.main()
    assert j1.read_text() == j2.read_text()


def test_cli_json_flags(tmp_path) -> None:
    inp = _write_midi(tmp_path)
    out = tmp_path / "o.mid"
    argv = [
        "sc",
        str(inp),
        "--out",
        str(out),
        "--pulse",
        "1/8",
        "--cycle-phrase-notes",
        "24",
        "--dry-run",
        "--legacy-bar-pulses-grid",
        "--section-lfo",
        '{"period":4}',
        "--stable-guard",
        '{"min_hold_beats":2}',
        "--vocal-adapt",
        '{"dense_onset":1,"dense_phrase":24,"sparse_phrase":24,"onsets":[0],"ratios":[0]}',
    ]

    def fake_pm(*a, **k):
        pm = _dummy_pm(2.0)
        if not a:
            pm.instruments = []
        return pm

    with mock.patch.object(sc.pretty_midi, "PrettyMIDI", side_effect=fake_pm):
        with mock.patch.object(sys, "argv", argv):
            sc.main()


def test_debug_md_header(tmp_path) -> None:
    stats = {
        "section_tags": {},
        "bar_phrase_notes_list": [],
        "fill_bars": [],
        "bar_reason": {},
        "lfo_pos": {},
        "guard_hold_beats": {},
        "bar_count": 0,
    }
    p = tmp_path / "d.md"
    sc.write_debug_md(stats, p)
    header = p.read_text().splitlines()[0]
    assert "lfo_pos" in header and "guard_hold_beats" in header


def test_damp_vocal_cli(tmp_path) -> None:
    inp = _write_midi(tmp_path)
    out = tmp_path / "out.mid"
    vcfg = (
        '{"dense_onset":0,"dense_phrase":24,"sparse_phrase":24,"onsets":[0,0],"ratios":[0.0,1.0]}'
    )
    argv = [
        "sc",
        str(inp),
        "--out",
        str(out),
        "--pulse",
        "1/8",
        "--cycle-phrase-notes",
        "24",
        "--vocal-adapt",
        vcfg,
        "--damp",
        "vocal:cc=11,channel=1",
    ]
    created = []

    def fake_pm(*a, **k):
        pm = _dummy_pm(2.0)
        if not a:
            pm.instruments = []
        created.append(pm)
        return pm

    with mock.patch.object(sc.pretty_midi, "PrettyMIDI", side_effect=fake_pm):
        with mock.patch.object(sys, "argv", argv):
            sc.main()
    out_pm = created[-1]
    inst = [i for i in out_pm.instruments if i.name == DAMP_INST_NAME]
    assert inst and inst[0].control_changes


def test_parse_json_arg_style_inject_error() -> None:
    with pytest.raises(SystemExit) as exc:
        sc.parse_json_arg(
            "--style-inject",
            "{\"period\":4,\"note\":\"A\",\"duration_beats\":0.5}",
            sc.STYLE_INJECT_SCHEMA,
        )
    assert "field 'note' must be int" in str(exc.value)


def test_parse_thresholds_missing_key() -> None:
    with pytest.raises(SystemExit) as exc:
        sc.parse_thresholds_arg("{\"low\":60,\"mid\":62}")
    assert "missing required key 'high'" in str(exc.value)


def test_parse_sections_invalid_json() -> None:
    with pytest.raises(SystemExit) as exc:
        sc.parse_json_arg("--sections", "{", sc.SECTIONS_SCHEMA)
    assert "expects JSON" in str(exc.value)


def test_parse_inline_chords_japanese_tokens() -> None:
    events = sc.parse_inline_chords("０：Ｃ♯：maj，２：Ｄ♭：min")
    assert events is not None
    assert len(events) == 2
    first = events[0]
    assert pytest.approx(first.start_beats or 0.0) == 0.0
    root_pc, quality = sc.parse_chord_symbol(first.chord)
    assert root_pc == sc.PITCH_CLASS["C#"]
    assert quality == "maj"


def test_inline_long_string_no_path_detection() -> None:
    spec = ",".join(f"{i}:C:maj" for i in range(0, 20, 2))
    events = sc.parse_inline_chords(spec)
    assert events is not None
    assert len(events) == 10


def test_marker_encoding_modes() -> None:
    class DummyPM:
        def __init__(self) -> None:
            self.markers = []

    def capture_markers(labels: list[str], mode: str) -> list[str]:
        dummy = DummyPM()
        sections = [
            {"start_bar": i, "end_bar": i + 1, "tag": label}
            for i, label in enumerate(labels)
        ]
        downbeats = [0.0, 1.0]
        with mock.patch.object(
            pretty_midi,
            "Marker",
            side_effect=lambda text, time: types.SimpleNamespace(text=text, time=time),
            create=True,
        ):
            sc._write_markers(dummy, True, sections, "intro", downbeats, mode)
        return [m.text for m in dummy.markers]

    assert capture_markers(["Intro🎵"], "ascii")[0] == "INTRO?"
    assert capture_markers(["✨"], "escape")[0] == "\\u2728"
    assert capture_markers(["a\u0308"], "escape")[0] == "\\u00c4"
    assert capture_markers(["サビ"], "raw")[0] == "サビ"


def test_marker_encoding_unknown_mode() -> None:
    dummy = types.SimpleNamespace(markers=[])
    with pytest.raises(SystemExit) as excinfo:
        sc._write_markers(dummy, True, [], "intro", [0.0, 1.0], "invalid")
    assert "unknown marker-encoding" in str(excinfo.value)


def test_stop_injection_requires_style_and_guides() -> None:
    inst = pretty_midi.Instrument(0)
    sc._inject_stops(inst, True, [(0.0, 1.0)], {0: None}, lambda b: b, {}, 0.5, 1.0, 90)
    assert len(inst.notes) == 0

    inst2 = pretty_midi.Instrument(0)
    sc._inject_stops(
        inst2,
        True,
        None,
        {},
        lambda b: b,
        {"style_stop": 32},
        0.5,
        1.0,
        90,
    )
    assert len(inst2.notes) == 0


def test_quantize_strength_pattern_across_bars() -> None:
    inst = pretty_midi.Instrument(0)
    notes = [
        pretty_midi.Note(velocity=90, pitch=60, start=0.12, end=0.4),
        pretty_midi.Note(velocity=90, pitch=60, start=0.63, end=0.9),
        pretty_midi.Note(velocity=90, pitch=60, start=1.12, end=1.4),
        pretty_midi.Note(velocity=90, pitch=60, start=1.63, end=1.9),
    ]
    inst.notes.extend(notes)
    sc._apply_quantize_safe(
        pretty_midi.PrettyMIDI(),
        inst,
        [1.0, 0.0],
        lambda b: b,
        lambda t: t,
        0.5,
        [0.0, 1.0, 2.0],
        None,
    )
    starts = [round(n.start, 2) for n in inst.notes]
    assert starts == [0.0, pytest.approx(0.63, rel=0.01), 1.0, pytest.approx(1.63, rel=0.01)]


def test_quicklook_summary_present() -> None:
    stats: dict[str, Any] = {"bar_density": {0: "low"}, "sections": ["intro"], "bar_count": 1}
    sc.finalize_phrase_track(
        pretty_midi.PrettyMIDI(),
        None,
        stats,
        {},
        section_lfo=None,
        lfo_targets=(),
        downbeats=[0.0, 1.0],
        guide_units=None,
        guide_units_time=None,
        guide_notes={},
        rest_ratios=None,
        onset_counts=None,
        chord_inst=None,
        phrase_inst=pretty_midi.Instrument(0),
        beat_to_time=lambda b: b,
        time_to_beat=lambda t: t,
        pulse_subdiv_beats=0.5,
        phrase_vel=90,
        phrase_merge_gap=0.0,
        release_sec=0.0,
        min_phrase_len_sec=0.0,
        stop_min_gap_beats=1.0,
        stop_velocity=90,
        damp_dst=None,
        damp_cc_num=0,
        guide_cc=None,
        bpm=120.0,
        section_overrides=None,
        fill_map=None,
        rest_silence_send_stop=False,
        quantize_strength=0.0,
        write_markers=False,
        marker_encoding="raw",
        section_labels=["intro"],
        section_default="intro",
        chord_merge_gap=0.01,
        clone_meta_only=False,
        meta_src="test",
        chords=None,
    )
    assert "quicklook" in stats
    quick = stats["quicklook"]
    assert quick["bar_count"] == 1
    assert quick["fill_count"] == 0
    assert quick["sections"] == ["intro"]


def test_meter_change_stats_consistency() -> None:
    class MeterPM:
        def __init__(self) -> None:
            self.instruments = [pretty_midi.Instrument(0)]
            self.time_signature_changes = [
                types.SimpleNamespace(numerator=6, denominator=8, time=0.0),
                types.SimpleNamespace(numerator=4, denominator=4, time=3.0),
            ]

        def get_beats(self):
            step = 0.5
            count = int(9.0 / step) + 1
            return [i * step for i in range(count)]

        def get_downbeats(self):
            return [0.0, 1.5, 3.0, 5.0, 7.0, 9.0]

        def get_end_time(self):
            return 9.0

        def get_tempo_changes(self):
            return [0.0], [120.0]

    pm = MeterPM()
    mapping = {
        "phrase_note": 36,
        "phrase_velocity": 96,
        "phrase_length_beats": 0.5,
        "cycle_phrase_notes": [36],
        "cycle_mode": "bar",
    }
    stats: dict[str, Any] = {"_legacy_bar_pulses_grid": False}
    chords = [sc.ChordSpan(0.0, 9.0, 0, "maj")]
    sc.build_sparkle_midi(
        pm,
        chords,
        mapping,
        0.5,
        "bar",
        0.0,
        0,
        "flat",
        120.0,
        0.0,
        0.5,
        stats=stats,
    )
    assert stats["schema_version"] == "1.1"
    assert stats["schema"] == "1.1"
    grid = stats["bar_pulse_grid"]
    expected = [
        sc.pulses_per_bar(6, 8, 0.5),
        sc.pulses_per_bar(6, 8, 0.5),
        sc.pulses_per_bar(4, 4, 0.5),
        sc.pulses_per_bar(4, 4, 0.5),
        sc.pulses_per_bar(4, 4, 0.5),
    ]
    for i, count in enumerate(expected):
        assert len(grid[i]) == count
    assert "bar_pulses" not in stats
    triggers = stats["bar_triggers"]
    assert stats["bar_trigger_pulses"] is triggers
    assert stats["bar_trigger_pulses_compat"] is triggers
    triggers = stats["bar_triggers"]
    actual_hits = sum(len(v) for v in triggers.values())
    assert actual_hits > 0


def test_meter_change_stats_legacy_mirror() -> None:
    class MeterPM:
        def __init__(self) -> None:
            self.instruments = [pretty_midi.Instrument(0)]
            self.time_signature_changes = [
                types.SimpleNamespace(numerator=6, denominator=8, time=0.0),
                types.SimpleNamespace(numerator=4, denominator=4, time=3.0),
            ]

        def get_beats(self):
            step = 0.5
            count = int(9.0 / step) + 1
            return [i * step for i in range(count)]

        def get_downbeats(self):
            return [0.0, 1.5, 3.0, 5.0, 7.0, 9.0]

        def get_end_time(self):
            return 9.0

        def get_tempo_changes(self):
            return [0.0], [120.0]

    pm = MeterPM()
    mapping = {
        "phrase_note": 36,
        "phrase_velocity": 96,
        "phrase_length_beats": 0.5,
        "cycle_phrase_notes": [36],
        "cycle_mode": "bar",
    }
    stats: Dict[str, Any] = {"_legacy_bar_pulses_grid": True}
    chords = [sc.ChordSpan(0.0, 9.0, 0, "maj")]
    sc.build_sparkle_midi(
        pm,
        chords,
        mapping,
        0.5,
        "bar",
        0.0,
        0,
        "flat",
        120.0,
        0.0,
        0.5,
        stats=stats,
    )
    assert stats["schema"] == "1.1"
    pulses = stats.get("bar_pulses")
    grid = stats["bar_pulse_grid"]
    assert pulses is not None
    assert pulses.keys() == grid.keys()
    for key, values in grid.items():
        assert pulses[key] == values
    assert stats["bar_trigger_pulses"] is stats["bar_triggers"]
