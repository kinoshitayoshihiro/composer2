from __future__ import annotations

"""
Vocal synchronization utilities for MIDI processing and timing analysis.

This module provides tools for:
- Loading vocal MIDI files
- Extracting note onsets and rest periods
- Loading consonant peaks from JSON
- Quantizing timing data to grids
- Converting time formats

Dependencies: pretty_midi, json, pathlib, decimal
"""

from pathlib import Path
from typing import Any, Iterable
import json
from decimal import Decimal

try:
    import pretty_midi
except ImportError:
    pretty_midi = None


def load_vocal_midi(path: str | Path) -> "pretty_midi.PrettyMIDI":
    """Load a MIDI file for vocal analysis.

    Parameters
    ----------
    path : str or Path
        Path to MIDI file.

    Returns
    -------
    pretty_midi.PrettyMIDI
        Loaded MIDI data.

    Raises
    ------
    ImportError
        If pretty_midi is not available.
    FileNotFoundError
        If file doesn't exist.
    """
    if pretty_midi is None:
        raise ImportError("pretty_midi is required for vocal MIDI loading")

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"MIDI file not found: {path}")

    return pretty_midi.PrettyMIDI(str(p))


def extract_onsets(
    pm: "pretty_midi.PrettyMIDI",
    *,
    tempo: float = 120.0,
    track_idx: int = 0,
    tempo_map: "pretty_midi.PrettyMIDI" | None = None,
) -> list[float]:
    """Extract note onsets from MIDI in beats.

    Parameters
    ----------
    pm : pretty_midi.PrettyMIDI
        MIDI object containing the vocal track.
    tempo : float, optional
        Tempo in BPM for conversion when ``tempo_map`` is ``None``.
    track_idx : int, optional
        Track index to analyse.
    tempo_map : pretty_midi.PrettyMIDI, optional
        External tempo map used to convert seconds to beats.

    Returns
    -------
    list[float]
        Onset times in beats.
    """
    if not pm.instruments:
        return []

    track = (
        pm.instruments[track_idx]
        if track_idx < len(pm.instruments)
        else pm.instruments[0]
    )

    if tempo_map is not None:
        onsets = [
            tempo_map.time_to_tick(note.start) / tempo_map.resolution
            for note in track.notes
        ]
    else:
        sec_per_beat = 60.0 / tempo
        onsets = [note.start / sec_per_beat for note in track.notes]

    return sorted(onsets)


def extract_long_rests(
    onsets: list[float],
    *,
    min_rest: float = 0.5,
    tempo_map: "pretty_midi.PrettyMIDI" | None = None,
    strict: bool | None = False,
) -> list[tuple[float, float]]:
    """Extract long rest periods between onsets.

    Parameters
    ----------
    onsets : list[float]
        Onset times. Interpreted as beats unless ``tempo_map`` is provided,
        in which case they are treated as seconds.
    min_rest : float, optional
        Minimum rest duration in beats.
    tempo_map : pretty_midi.PrettyMIDI, optional
        Tempo map for converting seconds to beats.
    strict : bool, optional
        When ``True`` and ``tempo_map`` is supplied, raise an error if the
        units appear ambiguous.

    Returns
    -------
    list[tuple[float, float]]
        Rest intervals. When ``tempo_map`` is ``None`` the pairs are
        ``(start_beat, end_beat)``; otherwise ``(start_beat, duration_beats)``.
    """
    if len(onsets) < 2:
        return []

    if tempo_map is not None:
        if strict and all(float(v).is_integer() for v in onsets):
            raise ValueError("ambiguous units")
        beats = [tempo_map.time_to_tick(t) / tempo_map.resolution for t in onsets]
    else:
        beats = list(onsets)

    rests = []
    for a, b in zip(beats, beats[1:]):
        dur = b - a
        if dur >= min_rest:
            if tempo_map is not None:
                rests.append((a, dur))
            else:
                rests.append((a, dur))

    return rests


def load_consonant_peaks(
    path: str | Path,
    *,
    tempo: float = 120.0,
    tempo_map: "pretty_midi.PrettyMIDI" | None = None,
) -> list[float]:
    """Load consonant peaks from JSON file.

    Parameters
    ----------
    path : str or Path
        Path to consonant peaks JSON file.
    tempo : float, optional
        Tempo in BPM for conversion when ``tempo_map`` is ``None``.
    tempo_map : pretty_midi.PrettyMIDI, optional
        Tempo map used to convert seconds to beats.

    Returns
    -------
    list[float]
        Peak times in beats.
    """
    path = Path(path)
    if not path.exists():
        return []

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    peaks = data.get("peaks", [])
    if tempo_map is not None:
        return sorted(tempo_map.time_to_tick(t) / tempo_map.resolution for t in peaks)

    sec_per_beat = 60.0 / float(tempo if tempo is not None else 120.0)
    return sorted(t / sec_per_beat for t in peaks)


def analyse_section(section: dict[str, Any], *, tempo_bpm: float) -> dict[str, list]:
    """Analyse a section and return vocal timing metrics.

    Parameters
    ----------
    section : dict
        Section configuration dictionary.
    tempo_bpm : float
        Tempo in beats per minute.

    Returns
    -------
    dict[str, list]
        Dictionary with ``"onsets"``, ``"rests"`` and ``"consonant_peaks"`` keys.
    """
    midi_path = section.get("vocal_midi_path")
    pm = load_vocal_midi(midi_path) if midi_path else None
    onsets = extract_onsets(pm, tempo=tempo_bpm) if pm else []
    rests = extract_long_rests(onsets, min_rest=0.5)
    peaks: list[float] = []
    cjson = section.get("consonant_json")
    if cjson:
        peaks = load_consonant_peaks(cjson, tempo=tempo_bpm)
    return {"onsets": onsets, "rests": rests, "consonant_peaks": peaks}


def quantize_times(
    times: Iterable[float],
    grid: float = 0.25,
    *,
    dedup: bool = False,
    eps: float = 1e-6,
    use_decimal: bool = False,
) -> list[float]:
    """Quantize times to grid size.

    Parameters
    ----------
    times : Iterable[float]
        Input time values in beats.
    grid : float, optional
        Quantization grid in beats.
    dedup : bool, optional
        If ``True``, remove duplicate quantized values.
    eps : float, optional
        Tolerance when comparing values.
    use_decimal : bool, optional
        Use ``decimal.Decimal`` for rounding.

    Returns
    -------
    list[float]
        Quantized times.

    Examples
    --------
    >>> quantize_times([0.1, 0.6, 1.1], grid=0.5)
    [0.0, 0.5, 1.0]
    """
    if use_decimal:
        grid_decimal = Decimal(str(grid))
        quantized = [
            float(
                (Decimal(str(t)) / grid_decimal).quantize(Decimal("1")) * grid_decimal
            )
            for t in times
        ]
    else:
        quantized = [round(t / grid) * grid for t in times]

    if dedup:
        seen = set()
        result = []
        for t in quantized:
            if not any(abs(t - s) < eps for s in seen):
                seen.add(t)
                result.append(t)
        return result

    return quantized
