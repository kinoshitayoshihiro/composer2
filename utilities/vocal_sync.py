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


def extract_onsets(
    pm: pretty_midi.PrettyMIDI,
    *,
    tempo: float | None = None,
    tempo_map: pretty_midi.PrettyMIDI | None = None,
) -> list[float]:
    """Return note onset times in beats.

    Parameters
    ----------
    pm : pretty_midi.PrettyMIDI
        Source MIDI whose note starts will be extracted.
    tempo : float, optional
        Fixed tempo in beats per minute. Ignored when ``tempo_map`` is given.
    tempo_map : pretty_midi.PrettyMIDI, optional
        MIDI containing tempo changes to use for time conversion. When omitted,
        ``pm`` itself is used for conversion.
        MIDI object to analyse.
    tempo : float, optional
        Fixed tempo in BPM if ``tempo_map`` is ``None``.
    tempo_map : pretty_midi.PrettyMIDI, optional
        Tempo map used for time conversion.

    Returns
    -------
    list[float]
        Note onsets in beats, sorted ascending.

    Examples
    --------
    >>> pm = pretty_midi.PrettyMIDI("v.mid")
    >>> extract_onsets(pm)[:2]
    [0.0, 0.5]
        Onset beats sorted ascending.

    Examples
    --------
    >>> pm = load_vocal_midi("v.mid")
    >>> extract_onsets(pm)[:2]
    [0.0, 1.0]
        Note onsets in beats, sorted ascending.
    """

    onsets: list[float] = []
    converter = tempo_map or pm
    sec_per_beat = 60.0 / tempo if tempo and not tempo_map else None
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            if sec_per_beat:
                beat = n.start / sec_per_beat
            else:
                beat = converter.time_to_tick(n.start) / converter.resolution
            onsets.append(float(beat))
    return sorted(onsets)


def extract_long_rests(
    onsets: Iterable[float],
    *,
    min_rest: float = 0.5,
    tempo: float | None = None,
    tempo_map: pretty_midi.PrettyMIDI | None = None,
    strict: bool = False,
) -> list[tuple[float, float]]:
    """Return long rests.

    Parameters
    ----------
    onsets : Iterable[float]
        Note onsets in beats or seconds.
    min_rest : float, optional
        Minimum rest length in beats.
    tempo : float, optional

        Tempo in BPM used when ``tempo_map`` is not given and ``onsets`` are
        in seconds.
    tempo_map : pretty_midi.PrettyMIDI, optional
        Tempo map that overrides ``tempo``.
        Fixed tempo in beats per minute. Used when ``tempo_map`` is ``None`` and
        ``onsets`` are given in seconds.
    tempo_map : pretty_midi.PrettyMIDI, optional
        Tempo map used to convert seconds to beats. Takes precedence over
        ``tempo``.
    strict : bool, optional
        If ``True`` and ``tempo_map`` is provided but ``onsets`` appear to be
        beats already (all values below 100 and divisible by 0.25), an
        exception is raised.

    Returns
    -------
    list[tuple[float, float]]
        Pairs of ``(start_beat, duration)`` for each long rest.

    Raises
    ------
    ValueError
        If ``strict`` is ``True`` and units appear ambiguous.
    """

    if strict and tempo_map is not None:
        grid = 0.25
        maybe_beats = all(
            t < 100 and abs((t / grid) - round(t / grid)) < 1e-6 for t in onsets
        )
        if maybe_beats:
            raise ValueError("ambiguous units")

    if tempo_map is not None:
        beats = [tempo_map.time_to_tick(t) / tempo_map.resolution for t in onsets]
    elif tempo is not None:
        sec_per_beat = 60.0 / tempo
        beats = [t / sec_per_beat for t in onsets]
    else:
        beats = list(onsets)

    rests: list[tuple[float, float]] = []
    sorted_onsets = sorted(beats)
    for a, b in zip(sorted_onsets, sorted_onsets[1:], strict=False):
        gap = b - a
        if gap > min_rest:
            rests.append((a, gap))
    return rests


def load_consonant_peaks(
    path: str | Path,
    *,
    tempo: float | None = 120.0,
    tempo_map: pretty_midi.PrettyMIDI | None = None,
) -> list[float]:
    """Load consonant peaks and convert to beats.

    Parameters
    ----------
    path : str or Path
        JSON file containing ``{"peaks": [seconds...]}``.
    tempo : float, optional
        Tempo in BPM used when ``tempo_map`` is not provided.
    tempo_map : pretty_midi.PrettyMIDI, optional
        MIDI used for tempo conversion when available.

    Returns
    -------
    list[float]
        Peak positions in beats, sorted.
        Sorted peak beats.

    Examples
    --------
    >>> load_consonant_peaks("p.json", tempo=100)
    [0.5, 1.0]
    """

    p = Path(path)
    if not p.exists():
        return []
    data = json.loads(p.read_text(encoding="utf-8"))
    peaks = [float(t) for t in data.get("peaks", [])]
    if tempo_map is not None:
        return sorted(tempo_map.time_to_tick(t) / tempo_map.resolution for t in peaks)
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


def quantize_times(
    times: Iterable[float], grid: float = 0.25, *, dedup: bool = False
) -> list[float]:
    """Quantize times to a grid in beats."""

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
