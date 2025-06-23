import json
from pathlib import Path
from statistics import median
from typing import List, Dict
from collections import deque
import math


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
            curve_type = str(e.get("curve", "linear"))
            events.append({"beat": beat, "bpm": bpm, "curve": curve_type})
        except (KeyError, TypeError, ValueError):
            continue
    events.sort(key=lambda x: x["beat"])
    return events


def _curve_fraction(t: float, mode: str) -> float:
    """Return interpolation fraction for ``mode`` at normalized ``t``."""

    if mode == "ease_in":
        return t * t
    if mode == "ease_out":
        return 1 - (1 - t) ** 2
    if mode == "ease_in_out":
        return 3 * t * t - 2 * t * t * t
    return t


def get_tempo_at_beat(beat: float, curve: List[Dict[str, float]]) -> float:
    """Return interpolated BPM at ``beat`` supporting multiple curves."""

    if not curve:
        return 120.0
    if beat <= curve[0]["beat"]:
        return float(curve[0]["bpm"])
    for i in range(1, len(curve)):
        prev = curve[i - 1]
        cur = curve[i]
        if beat <= cur["beat"]:
            mode = str(prev.get("curve", "linear"))
            if mode == "step":
                return float(prev["bpm"])
            span = cur["beat"] - prev["beat"]
            if span == 0:
                return float(cur["bpm"])
            t = (beat - prev["beat"]) / span
            frac = _curve_fraction(t, mode)
            return prev["bpm"] + (cur["bpm"] - prev["bpm"]) * frac
    return float(curve[-1]["bpm"])


def get_bpm_at(beat: float, curve: List[Dict[str, float]]) -> float:
    """Alias for :func:`get_tempo_at_beat`."""

    return get_tempo_at_beat(beat, curve)


def interpolate_bpm(curve: List[Dict[str, float]], beat: float) -> float:
    """Alias for :func:`get_bpm_at` for backward compatibility."""

    return get_bpm_at(beat, curve)


def beat_to_seconds(beat: float, curve: List[Dict[str, float]]) -> float:
    """Convert absolute beat position to seconds based on ``curve``."""
    if not curve:
        return beat * 0.5  # 120 BPM default

    curve = sorted(curve, key=lambda e: e["beat"])
    total = 0.0
    prev = curve[0]

    if beat <= prev["beat"]:
        return beat * 60.0 / prev["bpm"]

    def _seg_time(start_bpm: float, end_bpm: float, beats: float, mode: str) -> float:
        if beats <= 0:
            return 0.0
        if mode == "linear" and start_bpm != end_bpm:
            slope = (end_bpm - start_bpm) / beats
            return (60.0 / slope) * math.log(end_bpm / start_bpm)
        if mode == "step" or start_bpm == end_bpm:
            return beats * 60.0 / start_bpm

        # numeric integration for ease curves
        steps = 8
        total = 0.0
        for i in range(steps):
            t0 = i / steps
            t1 = (i + 1) / steps
            bpm0 = start_bpm + (end_bpm - start_bpm) * _curve_fraction(t0, mode)
            bpm1 = start_bpm + (end_bpm - start_bpm) * _curve_fraction(t1, mode)
            avg = (bpm0 + bpm1) / 2.0
            total += (t1 - t0) * beats * 60.0 / avg
        return total

    for cur in curve[1:]:
        if beat <= cur["beat"]:
            seg_beats = beat - prev["beat"]
            total += _seg_time(
                prev["bpm"], cur["bpm"], seg_beats, prev.get("curve", "linear")
            )
            return total
        seg_beats = cur["beat"] - prev["beat"]
        total += _seg_time(
            prev["bpm"], cur["bpm"], seg_beats, prev.get("curve", "linear")
        )
        prev = cur

    seg_beats = beat - prev["beat"]
    total += _seg_time(prev["bpm"], prev["bpm"], seg_beats, prev.get("curve", "linear"))
    return total


class TempoMap:
    """Lightweight tempo map supporting linear interpolation."""

    def __init__(self, events: List[Dict[str, float]]) -> None:
        cleaned: List[Dict[str, float]] = []
        for e in events:
            try:
                beat = float(e["beat"])
                bpm = float(e["bpm"])
            except (KeyError, TypeError, ValueError):
                continue
            cleaned.append({"beat": beat, "bpm": bpm})
        cleaned.sort(key=lambda x: x["beat"])
        if not cleaned:
            cleaned = [{"beat": 0.0, "bpm": 120.0}]
        self.events = cleaned

    def get_bpm(self, beat: float) -> float:
        curve = self.events
        if beat <= curve[0]["beat"]:
            return float(curve[0]["bpm"])
        for i in range(1, len(curve)):
            prev = curve[i - 1]
            cur = curve[i]
            if beat <= cur["beat"]:
                span = cur["beat"] - prev["beat"]
                if span <= 0:
                    return float(cur["bpm"])
                frac = (beat - prev["beat"]) / span
                return prev["bpm"] + (cur["bpm"] - prev["bpm"]) * frac
        return float(curve[-1]["bpm"])


def load_tempo_map(json_path: Path | str) -> TempoMap:
    try:
        with Path(json_path).open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        data = []
    if not isinstance(data, list):
        data = []
    return TempoMap(data)


class TempoVelocitySmoother:
    """MAD-based EMA smoother per instrument group."""

    def __init__(self, window: int = 8) -> None:
        if window <= 0:
            raise ValueError("window must be positive")
        self.window = window
        self.streams: Dict[str, Dict[str, any]] = {}

    def reset(self) -> None:
        for st in self.streams.values():
            st["hist"].clear()
            st["value"] = None

    @staticmethod
    def _group(inst: str) -> str | None:
        inst = inst.lower()
        if "kick" in inst or inst.startswith("bd"):
            return "kick"
        if "snare" in inst:
            return "snare"
        if "tom" in inst:
            return "tom"
        if any(k in inst for k in ["cymbal", "hh", "hat", "ride", "crash", "splash"]):
            return "cymbal"
        return None

    @staticmethod
    def _alpha(vals: List[int]) -> float:
        if not vals:
            return 0.5
        med = median(vals)
        dev = [abs(v - med) for v in vals]
        mad = median(dev)
        alpha = 0.1 + mad / 40.0
        if mad == 0 and max(vals) - min(vals) > 5:
            alpha = 0.3
        if alpha < 0.1:
            alpha = 0.1
        if alpha > 0.5:
            alpha = 0.5
        return alpha

    def smooth(self, inst: str, raw: int) -> int:
        grp = self._group(inst)
        if grp is None:
            return int(raw)
        stream = self.streams.setdefault(
            grp, {"hist": deque(maxlen=self.window), "value": None}
        )
        h: deque[int] = stream["hist"]
        h.append(int(raw))
        if stream["value"] is None:
            stream["value"] = float(raw)
            return int(raw)
        alpha = self._alpha(list(h))
        prev = stream["value"]
        new_val = alpha * raw + (1 - alpha) * prev
        stream["value"] = new_val
        out = int(round(new_val))
        return max(1, min(127, out))
