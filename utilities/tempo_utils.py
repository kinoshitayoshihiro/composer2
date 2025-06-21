import json
from pathlib import Path
from typing import List, Dict


def load_tempo_curve(path: Path) -> List[Dict[str, float]]:
    """Load tempo curve from JSON file.

    Each entry must contain ``beat`` and ``bpm`` fields. Invalid or
    malformed entries are ignored. On any read/parsing error an empty
    list is returned so callers can gracefully fall back to a constant
    tempo.
    """
    try:
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:  # pragma: no cover - optional safety
        return []
    if not isinstance(data, list):
        return []
    events = []
    for e in data:
        try:
            beat = float(e["beat"])
            bpm = float(e["bpm"])
            events.append({"beat": beat, "bpm": bpm})
        except (KeyError, TypeError, ValueError):
            continue
    events.sort(key=lambda x: x["beat"])
    return events


def get_bpm_at(beat: float, curve: List[Dict[str, float]]) -> float:
    """Return interpolated BPM at ``beat`` using linear interpolation."""
    if not curve:
        return 120.0
    if beat <= curve[0]["beat"]:
        return float(curve[0]["bpm"])
    for i in range(1, len(curve)):
        prev = curve[i - 1]
        cur = curve[i]
        if beat <= cur["beat"]:
            span = cur["beat"] - prev["beat"]
            if span == 0:
                return float(cur["bpm"])
            frac = (beat - prev["beat"]) / span
            return prev["bpm"] + (cur["bpm"] - prev["bpm"]) * frac
    return float(curve[-1]["bpm"])
