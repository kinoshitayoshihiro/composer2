import math
import sys
from pathlib import Path
from typing import List

import pytest

from tests import _stubs

sys.modules.setdefault("pretty_midi", _stubs.pretty_midi)

from ujam import sparkle_convert as sc
from ujam.consts import PITCH_CLASS


def _write_csv(tmp_path: Path, name: str, rows: List[str]) -> Path:
    path = tmp_path / name
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def test_read_chords_csv_full_format(tmp_path: Path) -> None:
    path = _write_csv(
        tmp_path,
        "explicit.csv",
        [
            "start,end,root,quality",
            "0.0,2.0,C,maj",
            "2.0,4.0,A,min",
        ],
    )
    spans = sc.read_chords_csv(path)
    assert len(spans) == 2
    assert spans[0] == sc.ChordSpan(0.0, 2.0, PITCH_CLASS["C"], "maj")
    assert spans[1] == sc.ChordSpan(2.0, 4.0, PITCH_CLASS["A"], "min")


def test_read_chords_csv_compact_header(tmp_path: Path) -> None:
    bar_times = [0.0, 2.0, 4.0, 6.0]
    meter_map = [(0.0, 4, 4)]
    path = _write_csv(
        tmp_path,
        "compact_header.csv",
        [
            "bar,chord",
            "0,C:maj",
            "1,D:min",
            "2,G",
        ],
    )
    spans = sc.read_chords_csv(
        path,
        bar_times=bar_times,
        meter_map=meter_map,
        bpm_hint=120.0,
    )
    assert [s.start for s in spans] == [0.0, 2.0, 4.0]
    assert [s.end for s in spans] == [2.0, 4.0, 6.0]
    assert spans[2].quality == "maj"


def test_read_chords_csv_compact_no_header(tmp_path: Path) -> None:
    bar_times = [0.0, 1.0, 2.0]
    path = _write_csv(tmp_path, "compact_no_header.csv", ["0,C", "1,Em"])
    spans = sc.read_chords_csv(path, bar_times=bar_times, bpm_hint=120.0)
    assert spans[0].quality == "maj"
    assert spans[1].quality == "min"
    assert spans[1].root_pc == PITCH_CLASS["E"]


def test_compact_chord_root_mapping(tmp_path: Path) -> None:
    tokens = [
        (0, "C"),
        (1, "C#"),
        (2, "Db"),
        (3, "D"),
        (4, "Eb"),
        (5, "E"),
        (6, "F"),
        (7, "F#"),
        (8, "Gb"),
        (9, "G"),
        (10, "Ab"),
        (11, "A"),
        (12, "Bb"),
        (13, "B"),
    ]
    path = _write_csv(
        tmp_path,
        "pc_map.csv",
        [f"{bar},{token}" for bar, token in tokens],
    )
    spans = sc.read_chords_csv(path, bpm_hint=90.0)
    pcs = [span.root_pc for span in spans]
    expected = [
        0,
        1,
        1,
        2,
        3,
        4,
        5,
        6,
        6,
        7,
        8,
        9,
        10,
        11,
    ]
    assert pcs == expected


def test_invalid_chord_symbol_raises(tmp_path: Path) -> None:
    path = _write_csv(tmp_path, "invalid.csv", ["0,??"])
    with pytest.raises(sc.ChordCsvError) as excinfo:
        sc.read_chords_csv(path, bpm_hint=100.0)
    assert "??" in str(excinfo.value)


def test_meter_map_respects_time_signature_changes(tmp_path: Path) -> None:
    bar_times = [0.0, 1.5, 3.0, 5.0, 7.0]
    meter_map = [(0.0, 6, 8), (3.0, 4, 4)]
    path = _write_csv(
        tmp_path,
        "meter_change.csv",
        [
            "bar,chord",
            "0,C",
            "1,G",
            "2,D",
            "3,C",
        ],
    )
    spans = sc.read_chords_csv(
        path,
        bar_times=bar_times,
        meter_map=meter_map,
        bpm_hint=120.0,
    )
    assert [round(s.start, 3) for s in spans] == [0.0, 1.5, 3.0, 5.0]
    assert [round(s.end, 3) for s in spans] == [1.5, 3.0, 5.0, 7.0]


def test_fallback_uses_bpm_hint(tmp_path: Path) -> None:
    path = _write_csv(
        tmp_path,
        "fallback.csv",
        [
            "bar,chord",
            "0,C",
            "1,F",
            "2,G",
        ],
    )
    spans = sc.read_chords_csv(path, bpm_hint=90.0)
    beat_seconds = 60.0 / 90.0
    bar_seconds = beat_seconds * 4.0
    assert math.isclose(spans[0].end - spans[0].start, bar_seconds)
    assert math.isclose(spans[1].start, bar_seconds)
    assert math.isclose(spans[-1].end, bar_seconds * 3)


def test_compact_with_bom_header(tmp_path: Path) -> None:
    path = tmp_path / "bom_header.csv"
    path.write_text("\ufeffbar,chord\n0,C\n1,G\n", encoding="utf-8")
    spans = sc.read_chords_csv(path, bpm_hint=120.0)
    assert [round(s.start, 3) for s in spans] == [0.0, 2.0]
    assert [round(s.end, 3) for s in spans] == [2.0, 4.0]


def test_compact_with_bom_no_header(tmp_path: Path) -> None:
    path = tmp_path / "bom_no_header.csv"
    path.write_text("\ufeff0,C\n1,G\n", encoding="utf-8")
    spans = sc.read_chords_csv(path, bpm_hint=120.0)
    assert len(spans) == 2
    assert spans[0].root_pc == PITCH_CLASS["C"]


def test_duplicate_bar_rejected(tmp_path: Path) -> None:
    path = _write_csv(tmp_path, "dup.csv", ["0,C", "0,G"])
    with pytest.raises(sc.ChordCsvError) as excinfo:
        sc.read_chords_csv(path, bpm_hint=120.0)
    assert "duplicate" in str(excinfo.value)


def test_descending_bar_rejected(tmp_path: Path) -> None:
    path = _write_csv(tmp_path, "desc.csv", ["1,C", "0,G"])
    with pytest.raises(sc.ChordCsvError) as excinfo:
        sc.read_chords_csv(path, bpm_hint=120.0)
    assert "ascending" in str(excinfo.value)


def test_negative_bar_rejected(tmp_path: Path) -> None:
    path = _write_csv(tmp_path, "neg.csv", ["-1,C"])
    with pytest.raises(sc.ChordCsvError) as excinfo:
        sc.read_chords_csv(path, bpm_hint=120.0)
    assert "bar must be >= 0" in str(excinfo.value)


def test_multi_bar_span(tmp_path: Path) -> None:
    path = _write_csv(
        tmp_path,
        "multi.csv",
        [
            "bar_start,bar_end,chord",
            "0,2,C",
            "2,3,G",
        ],
    )
    spans = sc.read_chords_csv(path, bpm_hint=120.0)
    assert len(spans) == 2
    assert math.isclose(spans[0].end - spans[0].start, 4.0)
    assert math.isclose(spans[1].start, 4.0)


def test_bar_beat_chord_format(tmp_path: Path) -> None:
    path = _write_csv(
        tmp_path,
        "beats.csv",
        [
            "bar,beat,chord",
            "0,0,C",
            "0,2,G7",
            "1,1/2,Dm",
        ],
    )
    spans = sc.read_chords_csv(path, bpm_hint=120.0)
    assert len(spans) == 3
    starts = [round(s.start, 3) for s in spans]
    ends = [round(s.end, 3) for s in spans]
    assert starts == [0.0, 1.0, 2.25]
    assert ends[-1] == 4.0
    assert spans[1].quality == "7"


def test_extended_quality_aliases(tmp_path: Path) -> None:
    rows = [
        "0,G7",
        "1,Cmaj7",
        "2,Dm7b5",
        "3,Fsus4/G",
        "4,Aadd9",
    ]
    path = _write_csv(tmp_path, "qualities.csv", rows)
    spans = sc.read_chords_csv(path, bpm_hint=100.0)
    qualities = [s.quality for s in spans]
    assert qualities == ["7", "maj7", "m7b5", "sus4/G", "add9"]
    symbols = [getattr(s, "symbol", None) for s in spans]
    assert symbols == ["G7", "Cmaj7", "Dm7b5", "Fsus4/G", "Aadd9"]
    roots = [getattr(s, "root_name", None) for s in spans]
    assert roots == ["G", "C", "D", "F", "A"]


def test_meter_map_with_in_bar_positions(tmp_path: Path) -> None:
    bar_times = [0.0, 1.5, 3.0, 5.0]
    meter_map = [(0.0, 6, 8), (3.0, 4, 4)]
    path = _write_csv(
        tmp_path,
        "meter_beat.csv",
        [
            "bar,beat,chord",
            "0,3,C",
            "1,0,G",
            "2,2,D",
        ],
    )
    spans = sc.read_chords_csv(
        path,
        bar_times=bar_times,
        meter_map=meter_map,
        bpm_hint=120.0,
    )
    starts = [round(s.start, 3) for s in spans]
    assert starts[0] == 0.75  # half-way through 6/8 bar
    assert starts[1] == 1.5
    assert starts[2] == 4.0


def test_fallback_non_four_four_default_meter(tmp_path: Path) -> None:
    path = _write_csv(tmp_path, "waltz.csv", ["0,C", "1,G"])
    spans = sc.read_chords_csv(path, bpm_hint=60.0, default_meter=(3, 4))
    assert math.isclose(spans[0].end - spans[0].start, 3.0)
    assert math.isclose(spans[1].start, 3.0)


def test_explicit_chord_column_case_insensitive(tmp_path: Path) -> None:
    path = _write_csv(
        tmp_path,
        "explicit_chord_col.csv",
        [
            "Start,End,Chord",
            "0.0,1.5,Cmaj7",
            "1.5,3.0,dm7",
        ],
    )
    spans = sc.read_chords_csv(path)
    assert [s.quality for s in spans] == ["maj7", "m7"]
    assert [getattr(s, "symbol", None) for s in spans] == ["Cmaj7", "dm7"]
    assert [getattr(s, "root_name", None) for s in spans] == ["C", "d"]


def test_compact_header_case_insensitive(tmp_path: Path) -> None:
    path = _write_csv(
        tmp_path,
        "case_header.csv",
        [
            "Bar,Chord",
            "0,C",
            "1,G",
        ],
    )
    spans = sc.read_chords_csv(path, bpm_hint=120.0)
    assert [round(s.start, 3) for s in spans] == [0.0, 2.0]
    assert [round(s.end, 3) for s in spans] == [2.0, 4.0]


def test_compact_prefers_beat_grid_over_bpm(tmp_path: Path) -> None:
    path = _write_csv(
        tmp_path,
        "tempo_change.csv",
        [
            "bar,chord",
            "0,C",
            "1,G",
        ],
    )
    beat_times = [0.0, 0.5, 1.0, 1.5, 2.0, 2.75, 3.5, 4.25, 5.0]
    spans = sc.read_chords_csv(
        path,
        beat_times=beat_times,
        meter_map=[(0.0, 4, 4)],
        bpm_hint=90.0,
    )
    assert math.isclose(spans[0].end, 2.0)
    assert math.isclose(spans[1].start, 2.0)
    assert math.isclose(spans[1].end, 5.0)


def test_compact_meter_hints_override_default(tmp_path: Path) -> None:
    path = _write_csv(
        tmp_path,
        "meter_hints.csv",
        [
            "bar,chord",
            "0,C",
            "1,G",
            "2,D",
        ],
    )
    spans = sc.read_chords_csv(
        path,
        meter_hints=[(0, 12, 8), (2, 4, 4)],
        bpm_hint=60.0,
        default_meter=(4, 4),
    )
    starts = [round(s.start, 3) for s in spans]
    ends = [round(s.end, 3) for s in spans]
    assert starts == [0.0, 6.0, 12.0]
    assert ends == [6.0, 12.0, 16.0]


def test_strict_mode_reports_line_numbers(tmp_path: Path) -> None:
    path = _write_csv(
        tmp_path,
        "overlap.csv",
        [
            "start,end,root,quality",
            "0,2,C,maj",
            "1,3,G,maj",
        ],
    )
    with pytest.raises(sc.ChordCsvError) as excinfo:
        sc.read_chords_csv(path, strict=True)
    assert "line 3" in str(excinfo.value)

