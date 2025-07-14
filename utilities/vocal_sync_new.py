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

    Args:
        path: Path to MIDI file

    Returns:
        PrettyMIDI object

    Raises:
        ImportError: If pretty_midi is not available
        FileNotFoundError: If file doesn't exist
    """
    if pretty_midi is None:
        raise ImportError("pretty_midi is required for vocal MIDI loading")

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"MIDI file not found: {path}")

    return pretty_midi.PrettyMIDI(str(p))


def extract_onsets(
    pm: "pretty_midi.PrettyMIDI", *, tempo: float = 120.0, track_idx: int = 0
) -> list[float]:
    """Extract note onsets from MIDI in beats.

    Args:
        pm: PrettyMIDI object
        tempo: Tempo in BPM for beat conversion
        track_idx: Track index to analyze

    Returns:
        List of onset times in beats
    """
    if not pm.instruments:
        return []

    track = (
        pm.instruments[track_idx]
        if track_idx < len(pm.instruments)
        else pm.instruments[0]
    )

    # Convert seconds to beats: seconds * (tempo/60) = beats
    sec_per_beat = 60.0 / tempo
    onsets = [note.start / sec_per_beat for note in track.notes]

    return sorted(onsets)


def extract_long_rests(
    onsets: list[float], *, min_rest: float = 0.5
) -> list[tuple[float, float]]:
    """Extract long rest periods between onsets.

    Args:
        onsets: List of onset times in beats
        min_rest: Minimum rest duration to include

    Returns:
        List of (start, end) tuples for rest periods
    """
    if len(onsets) < 2:
        return []

    rests = []
    for i in range(1, len(onsets)):
        rest_start = onsets[i - 1]
        rest_end = onsets[i]
        rest_duration = rest_end - rest_start

        if rest_duration >= min_rest:
            rests.append((rest_start, rest_end))

    return rests


def load_consonant_peaks(
    path: str | Path, *, tempo: float = 120.0, tempo_map=None
) -> list[float]:
    """Load consonant peaks from JSON file.

    Args:
        path: Path to consonant peaks JSON file
        tempo: Tempo in BPM for conversion
        tempo_map: Optional tempo map for conversion

    Returns:
        List of peak times in beats
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

    Args:
        section: Section configuration dictionary
        tempo_bpm: Tempo in beats per minute

    Returns:
        Dictionary with 'onsets', 'rests', and 'consonant_peaks' keys
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

    Args:
        times: Input time values in beats
        grid: Quantization grid in beats
        dedup: If True, remove duplicate quantized values
        eps: Tolerance when comparing values
        use_decimal: Use decimal.Decimal for rounding

    Returns:
        List of quantized times

    Examples:
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
