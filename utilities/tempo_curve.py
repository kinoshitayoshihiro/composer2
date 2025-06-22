from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from bisect import bisect_right
import json
import math
from typing import Iterable, List

import yaml  # type: ignore
from music21 import meter

@dataclass
class TempoPoint:
    beat: float
    bpm: float

class TempoCurve:
    """Time-varying tempo curve with linear interpolation."""

    def __init__(self, points: Iterable[dict]) -> None:
        pts: List[TempoPoint] = []
        for item in points:
            try:
                beat = float(item["beat"])
                bpm = float(item["bpm"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("Invalid tempo curve entry") from exc
            if bpm <= 0 or math.isnan(bpm) or math.isinf(bpm):
                raise ValueError("Invalid BPM value")
            pts.append(TempoPoint(beat, bpm))

        if not pts:
            raise ValueError("TempoCurve requires at least one point")

        pts.sort(key=lambda p: p.beat)
        beats = [p.beat for p in pts]
        if len(beats) != len(set(beats)):
            raise ValueError("Duplicate beat entry in tempo curve")

        self.points = pts
        self.beats = beats

    @classmethod
    def from_json(cls, path: str | Path) -> "TempoCurve":
        p = Path(path)
        with p.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, list):
            raise ValueError("Invalid tempo curve format")
        return cls(data)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TempoCurve":
        p = Path(path)
        with p.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        if not isinstance(data, list):
            raise ValueError("Invalid tempo curve format")
        return cls(data)

    def bpm_at(
        self, value: float, ts: meter.TimeSignature | None = None
    ) -> float:
        """Return BPM at an absolute offset or beat."""

        beat = (
            value / ts.beatDuration.quarterLength if ts is not None else value
        )

        pts = self.points
        if beat <= pts[0].beat:
            return pts[0].bpm
        if beat >= pts[-1].beat:
            return pts[-1].bpm
        idx = bisect_right(self.beats, beat)
        prev_pt = pts[idx - 1]
        next_pt = pts[idx]
        span = next_pt.beat - prev_pt.beat
        if span <= 0:
            return next_pt.bpm
        frac = (beat - prev_pt.beat) / span
        return prev_pt.bpm + (next_pt.bpm - prev_pt.bpm) * frac

    def spb_at(self, beat: float) -> float:
        bpm = self.bpm_at(beat)
        return 60.0 / bpm


def load_tempo_curve(path: Path | str) -> TempoCurve:
    """Load :class:`TempoCurve` from ``path`` auto-detecting JSON or YAML."""

    p = Path(path)
    if p.suffix.lower() in {".yml", ".yaml"}:
        return TempoCurve.from_yaml(p)
    return TempoCurve.from_json(p)
