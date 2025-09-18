import argparse
from pathlib import Path
from typing import List

import pytest

try:
    import pretty_midi  # type: ignore
except Exception:  # pragma: no cover
    from tests._stubs import pretty_midi  # type: ignore

from ujam import sparkle_convert as sc
from ujam.consts import PHRASE_INST_NAME


def _ensure_marker_class() -> None:
    if hasattr(pretty_midi, "Marker"):
        return

    class _Marker:  # pragma: no cover - fallback for stripped pretty_midi
        def __init__(self, text: str, time: float) -> None:
            self.text = text
            self.time = time

    pretty_midi.Marker = _Marker  # type: ignore[attr-defined]


def _bar_times(count: int, step: float = 1.0) -> List[float]:
    return [i * step for i in range(count)]


def test_cli_compact_chords_csv_variants(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SPARKLE_DETERMINISTIC", "1")
    two_col = tmp_path / "compact2.csv"
    two_col.write_text("0,C:maj\n2,F:min\n", encoding="utf-8")
    bar_times = _bar_times(6)
    spans_two = sc.read_chords_csv(two_col, bar_times=bar_times)
    assert [(round(span.start, 2), round(span.end, 2), span.quality) for span in spans_two] == [
        (0.0, 2.0, "maj"),
        (2.0, 3.0, "min"),
    ]
    assert [span.root_pc for span in spans_two] == [0, 5]

    three_col = tmp_path / "compact3.csv"
    three_col.write_text("bar,beat,chord\n0,0,C:maj\n3,2,G:min\n", encoding="utf-8")
    spans_three = sc.read_chords_csv(three_col, bar_times=bar_times)
    assert [(round(span.start, 2), round(span.end, 2), span.quality) for span in spans_three] == [
        (0.0, 3.5, "maj"),
        (3.5, 4.5, "min"),
    ]
    assert [span.root_pc for span in spans_three] == [0, 7]


def test_cli_compact_csv_error_excerpt(tmp_path: Path) -> None:
    bad = tmp_path / "bad.csv"
    bad.write_text("bar,chord\n0,BadSymbol\n1,C:maj\n", encoding="utf-8")
    with pytest.raises(SystemExit) as excinfo:
        sc.read_chords_csv(bad, bar_times=[0.0, 1.0, 2.0])
    msg = str(excinfo.value)
    assert "token 0" in msg
    assert "BadSymbol" in msg


def test_cli_section_end_with_mixed_sections(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SPARKLE_DETERMINISTIC", "1")
    class DummyPM:
        def __init__(self) -> None:
            inst = pretty_midi.Instrument(0, name=PHRASE_INST_NAME)
            inst.is_drum = False
            self.instruments = [inst]
            self.time_signature_changes = []
            self.markers = []  # type: ignore[attr-defined]

        def get_beats(self) -> List[float]:
            return [i * 0.5 for i in range(8)]

        def get_downbeats(self) -> List[float]:
            return [0.0, 1.0, 2.0, 3.0]

        def get_end_time(self) -> float:
            return 3.0

        def get_tempo_changes(self):  # pragma: no cover - unused by assertion
            return [0.0], [120.0]

    pm = DummyPM()
    phrase_inst = pm.instruments[0]
    downbeats = pm.get_downbeats()
    units = [(downbeats[i], downbeats[i + 1]) for i in range(len(downbeats) - 1)]
    stats = {"downbeats": list(downbeats), "sections": ["Intro", "Intro", "Outro"]}
    setattr(pm, "_sparkle_stats", stats)
    mapping = {
        "phrase_note": 36,
        "phrase_velocity": 96,
        "cycle_phrase_notes": [],
        "style_fill": 34,
    }
    args = argparse.Namespace(
        auto_fill="section_end",
        fill_length_beats=0.5,
        fill_min_gap_beats=0.0,
        fill_avoid_pitches=None,
        guide_rest_silence_th=0.75,
    )
    sc.finalize_phrase_track(
        pm,
        args,
        stats,
        mapping,
        downbeats=downbeats,
        guide_units=units,
        guide_units_time=units,
        rest_ratios=[0.0, 0.0, 0.0],
        onset_counts=[0, 0, 0],
        section_overrides=["Intro", {"start_bar": 2, "tag": "Outro"}],
        section_default="verse",
        write_markers=False,
        marker_encoding="raw",
        phrase_inst=phrase_inst,
    )
    fills = stats.get("fills") or []
    assert fills, "expected section_end auto-fill to insert notes"
    bar_indices = {fill["bar"] for fill in fills}
    assert bar_indices <= {1, 2}
    assert all(note.pitch != mapping["phrase_note"] for note in phrase_inst.notes if note.start >= 1.0)


def test_cli_marker_encoding_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SPARKLE_DETERMINISTIC", "1")
    _ensure_marker_class()
    pm = pretty_midi.PrettyMIDI()
    pm.markers = []  # type: ignore[attr-defined]
    phrase_inst = pretty_midi.Instrument(program=0, name=PHRASE_INST_NAME)
    pm.instruments.append(phrase_inst)
    downbeats = [0.0, 1.0, 2.0]
    stats = {"downbeats": list(downbeats), "sections": ["あ", "B"]}
    mapping = {"phrase_note": 36, "phrase_velocity": 96, "cycle_phrase_notes": []}
    args = argparse.Namespace(auto_fill="off")
    sc.finalize_phrase_track(
        pm,
        args,
        stats,
        mapping,
        downbeats=downbeats,
        guide_units=[(0.0, 1.0), (1.0, 2.0)],
        guide_units_time=[(0.0, 1.0), (1.0, 2.0)],
        rest_ratios=[0.0, 0.0],
        onset_counts=[0, 0],
        section_overrides=["あ", "B"],
        section_default="verse",
        write_markers=True,
        marker_encoding="ASCII",
    )
    assert pm.markers  # type: ignore[attr-defined]
    assert pm.markers[0].text == "?"  # type: ignore[index]
