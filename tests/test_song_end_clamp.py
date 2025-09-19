import math

try:  # pragma: no cover - optional dependency
    import pretty_midi  # type: ignore  # noqa: F401
except Exception:  # pragma: no cover - fallback stub
    from tests import _stubs as pretty_midi  # type: ignore  # noqa: F401

import math
import tempfile
from pathlib import Path

import ujam.sparkle_convert as sc


def _pm_44(bpm: float = 71.4):
    midi = sc.pretty_midi.PrettyMIDI(initial_tempo=bpm)
    ts_cls = getattr(sc.pretty_midi, "TimeSignature", None)
    if ts_cls is not None:
        try:
            midi.time_signature_changes = [ts_cls(4, 4, 0.0)]
        except Exception:
            pass
    if not hasattr(midi, "get_beats"):
        beat_len = 60.0 / bpm if bpm > 0 else 0.5
        beats = [i * beat_len for i in range(512)]

        def _beats() -> list[float]:
            return list(beats)

        midi.get_beats = _beats  # type: ignore[attr-defined]
    if not hasattr(midi, "get_end_time"):
        midi.get_end_time = lambda: 0.0  # type: ignore[attr-defined]
    return midi


def _bar_seconds(bpm: float) -> float:
    return 4 * 60.0 / bpm


def _basic_mapping() -> dict:
    return {
        "phrase_note": 36,
        "phrase_velocity": 100,
        "phrase_length_beats": 0.5,
        "cycle_phrase_notes": [36],
        "cycle_mode": "bar",
    }


def _pm_end_time(pm) -> float:
    end = 0.0
    for inst in getattr(pm, "instruments", []):
        for note in getattr(inst, "notes", []):
            end = max(end, getattr(note, "end", 0.0))
    return end


def test_sections_force_end_trim():
    bpm = 71.4
    midi = _pm_44(bpm)
    end_time = 84 * _bar_seconds(bpm)
    chords = [sc.ChordSpan(0.0, end_time, 0, "maj")]
    mapping = _basic_mapping()
    stats = {}
    out = sc.build_sparkle_midi(
        midi,
        chords,
        mapping,
        0.5,
        "bar",
        0.0,
        0,
        "flat",
        bpm,
        0.0,
        0.5,
        stats=stats,
        sections=[{"start_bar": 0, "end_bar": 84, "tag": "all"}],
    )
    out_end = stats.get("song_end_time")
    assert math.isclose(out_end or 0.0, end_time, abs_tol=1e-6)
    for inst in out.instruments:
        for note in inst.notes:
            assert note.end <= (out_end or 0.0) + 1e-6
    assert stats.get("notes_clipped", 0) >= 0


def test_style_inject_alias_name():
    bpm = 71.4
    midi = _pm_44(bpm)
    end_time = 16 * _bar_seconds(bpm)
    chords = [sc.ChordSpan(0.0, end_time, 0, "maj")]
    mapping = _basic_mapping()
    mapping["phrase_aliases"] = {"open_1_4": 39}
    mapping["phrase_pool"] = {"pool": [{"name": "open_1_4", "note": 39, "weight": 1.0}]}
    mapping["style_inject"] = sc.validate_style_inject_cfg(
        {"period": 4, "note": "open_1_4", "duration_beats": 0.5}, mapping
    )
    out = sc.build_sparkle_midi(
        midi,
        chords,
        mapping,
        0.5,
        "bar",
        0.0,
        0,
        "flat",
        bpm,
        0.0,
        0.5,
        fill_policy="style",
    )
    pitches = [n.pitch for inst in out.instruments for n in inst.notes]
    assert 39 in pitches


def test_chords_csv_compact(tmp_path: Path):
    two_col = tmp_path / "two.csv"
    two_col.write_text("bar,chord\n0,C:maj\n2,G:maj\n3,Am:min\n", encoding="utf-8")
    spans = sc.read_chords_csv(two_col, bpm_hint=120.0, default_meter=(4, 4))
    assert [round(s.start, 3) for s in spans] == [0.0, 4.0, 6.0]

    three_col = tmp_path / "three.csv"
    three_col.write_text("bar,chord,beats\n0,C:maj,2\n1,G:maj,4\n2,Am:min\n", encoding="utf-8")
    spans_three = sc.read_chords_csv(three_col, bpm_hint=120.0, default_meter=(4, 4))
    assert len(spans_three) == 3
    assert math.isclose(spans_three[0].end - spans_three[0].start, _bar_seconds(120.0) / 2, rel_tol=0, abs_tol=1e-6)


def test_mido_write_types(tmp_path: Path):
    bpm = 90.0
    midi = _pm_44(bpm)
    chords = [sc.ChordSpan(0.0, 8 * _bar_seconds(bpm), 0, "maj")]
    mapping = _basic_mapping()
    out = sc.build_sparkle_midi(
        midi,
        chords,
        mapping,
        0.5,
        "bar",
        0.0,
        0,
        "flat",
        bpm,
        0.0,
        0.5,
    )
    with tempfile.NamedTemporaryFile(delete=False) as handle:
        path = Path(handle.name)
    try:
        out.write(str(path))
    finally:
        path.unlink(missing_ok=True)

