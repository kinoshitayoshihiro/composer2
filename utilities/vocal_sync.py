"""Vocal-driven timing utilities."""

from __future__ import annotations

import json
from collections.abc import Iterable
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

try:
    import pretty_midi
except ImportError as e:  # pragma: no cover
    raise ImportError("pretty_midi is required") from e

__all__ = [
    "load_vocal_midi",
    "extract_onsets",
    "extract_long_rests",
    "load_consonant_peaks",
    "analyse_section",
    "quantize_times",
]


def load_vocal_midi(path: str | Path) -> pretty_midi.PrettyMIDI:
    """Load a MIDI file containing the vocal line."""

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    return pretty_midi.PrettyMIDI(str(p))

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    return pretty_midi.PrettyMIDI(str(p))
    pretty_midi = None


def load_vocal_midi(path: str | Path) -> pretty_midi.PrettyMIDI:
    """Load a vocal MIDI file.

    Parameters
    ----------
    path : str or Path
        Path to the MIDI file.

    Returns
    -------
    pretty_midi.PrettyMIDI
        Loaded MIDI data.
    """
    if pretty_midi is None:
        raise ImportError("pretty_midi is required for vocal sync functionality")
    return pretty_midi.PrettyMIDI(str(path))


def extract_onsets(
    pm: pretty_midi.PrettyMIDI,
    *,
    tempo_map: pretty_midi.PrettyMIDI | None = None,
) -> list[float]:
    """Return note onset times in beats.

    Parameters
    ----------
    pm : pretty_midi.PrettyMIDI
        MIDI data containing vocal notes.
    tempo_map : pretty_midi.PrettyMIDI, optional
        Tempo map for beat conversion.

    Returns
    -------
    list[float]
        Note onset times in beats.
    """
    onsets = []
    for instrument in pm.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            if tempo_map:
                beat_time = tempo_map.time_to_tick(note.start) / tempo_map.resolution
            else:
                beat_time = pm.time_to_tick(note.start) / pm.resolution
            onsets.append(beat_time)
    return sorted(onsets)


def find_long_rests(
    onsets: list[float],
    *,
    min_rest: float = 2.0,
) -> list[tuple[float, float]]:
    """Return long rests.

    Parameters
    ----------
    onsets : list[float]
        Note onset times in beats.
    min_rest : float, optional
        Minimum rest duration in beats.

    Returns
    -------
    list[tuple[float, float]]
        Rest intervals as (start, end) pairs.
    """
    if not onsets:
        return []

    rests = []
    for i in range(len(onsets) - 1):
        rest_duration = onsets[i + 1] - onsets[i]
        if rest_duration >= min_rest:
            rests.append((onsets[i], onsets[i + 1]))
    return rests


def extract_long_rests(
    onsets: list[float],
    *,
    min_rest: float = 2.0,
    tempo_map: pretty_midi.PrettyMIDI | None = None,
    strict: bool | None = False,
) -> list[tuple[float, float]]:
    """Return long rests from onsets.

    Parameters
    ----------
    onsets : list[float]
        Note onsets in beats if ``tempo_map`` is ``None``. Otherwise they are
        interpreted as seconds.
    min_rest : float, optional
        Minimum rest duration in beats.
    tempo_map : pretty_midi.PrettyMIDI, optional
        Tempo map used to convert seconds to beats. If omitted, ``onsets`` are
        assumed to be in beats.
    strict : bool, optional
        If ``True`` and ``tempo_map`` is provided, raise :class:`ValueError` if
        the units of ``onsets`` appear ambiguous.

    Returns
    -------
    list[tuple[float, float]]
        Long rests represented as ``(start_beat, duration_beats)`` tuples.
    """

    if not onsets:
        return []

    if tempo_map is not None:
        if strict and all(float(v).is_integer() for v in onsets):
            raise ValueError("ambiguous units")
        beats = [tempo_map.time_to_tick(t) / tempo_map.resolution for t in onsets]
    else:
        beats = list(onsets)

    rests: list[tuple[float, float]] = []
    for a, b in zip(beats, beats[1:]):
        dur = b - a
        if dur >= min_rest:
            rests.append((a, dur))
    return rests


def load_consonant_peaks(
    path: str | Path,
    *,
    tempo_map: pretty_midi.PrettyMIDI | None = None,
    tempo: float = 120.0,
) -> list[float]:
    """Load consonant peaks and convert to beats.

    Parameters
    ----------
    path : str or Path
        Path to JSON file containing consonant peaks.

    Returns
    -------
    list[float]
        Consonant peak times in beats.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    peaks = data.get("peaks", [])
    if tempo_map is not None:
<       return sorted(tempo_map.time_to_tick(t) / tempo_map.resolution for t in peaks)
    sec_per_beat = 60.0 / float(tempo if tempo is not None else 120.0)
    return sorted(t / sec_per_beat for t in peaks)


def analyse_section(section: dict[str, Any], *, tempo_bpm: float) -> dict[str, list]:
    """Analyse a section and return vocal timing metrics."""

    midi_path = section.get("vocal_midi_path")
    pm = load_vocal_midi(midi_path) if midi_path else None
    onsets = extract_onsets(pm, tempo=tempo_bpm) if pm else []
    rests = extract_long_rests(onsets, min_rest=0.5)
    peaks: list[float] = []
    cjson = section.get("consonant_json")
    if cjson:
        peaks = load_consonant_peaks(cjson, tempo=tempo_bpm)
    return {"onsets": onsets, "rests": rests, "consonant_peaks": peaks}
        return [tempo_map.time_to_tick(t) / tempo_map.resolution for t in peaks]
    return [t * tempo / 60.0 for t in peaks]


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
    grid : float
        Quantization grid in beats.
    dedup : bool, optional
        If ``True``, remove duplicate quantized values.
    eps : float, optional
        Tolerance when comparing values.
    use_decimal : bool, optional
        Use ``decimal.Decimal`` for rounding.
def quantize_times(
    times: Iterable[float], grid: float = 0.25, *, dedup: bool = False
) -> list[float]:
    """Quantize times to a grid in beats."""

    q = [round(t / grid) * grid for t in times]
    Examples
    --------
    >>> quantize_times([0.1, 0.26], 0.25)
    [0.0, 0.25]
    """
    if use_decimal:
        factor = Decimal(str(grid))
        q = [
            float((Decimal(str(t)) / factor).to_integral_value(ROUND_HALF_UP) * factor)
            for t in times
        ]
    else:
        q = [round(t / grid) * grid for t in times]

    if dedup:
        out: list[float] = []
        for v in sorted(q):
            if not out or abs(v - out[-1]) > 1e-6:
                out.append(v)
        return out
    return q


if __name__ == "__main__":  # pragma: no cover - simple CLI
    import argparse

    parser = argparse.ArgumentParser(description="Print vocal onsets from MIDI")
    parser.add_argument("midi")
    parser.add_argument("--tempo-map", dest="tempo_map")
    args = parser.parse_args()

    pm = load_vocal_midi(args.midi)
    tempo_map = load_vocal_midi(args.tempo_map) if args.tempo_map else None
    for o in extract_onsets(pm, tempo_map=tempo_map):
        print(o)
