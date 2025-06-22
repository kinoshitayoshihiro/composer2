from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from bisect import bisect_right
import json
import math
from typing import List

@dataclass
class TempoPoint:
    beat: float
    bpm: float

class TempoCurve:
    def __init__(self, points: List[TempoPoint]) -> None:
        if not points:
            raise ValueError("TempoCurve requires at least one point")
        points = sorted(points, key=lambda p: p.beat)
        seen = set()
        for pt in points:
            if pt.beat in seen:
                raise ValueError("Duplicate beat entry in tempo curve")
            seen.add(pt.beat)
            if pt.bpm <= 0 or math.isnan(pt.bpm) or math.isinf(pt.bpm):
                raise ValueError("Invalid BPM value")
        self.points = points
        self.beats = [p.beat for p in points]

    @classmethod
    def from_json(cls, path: str | Path) -> "TempoCurve":
        p = Path(path)
        with p.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, list):
            raise ValueError("Invalid tempo curve format")
        points: List[TempoPoint] = []
        for item in data:
            try:
                beat = float(item["beat"])
                bpm = float(item["bpm"])
            except (KeyError, TypeError, ValueError) as e:
                raise ValueError("Invalid tempo curve entry") from e
            points.append(TempoPoint(beat, bpm))
        return cls(points)

    def bpm_at(self, beat: float) -> float:
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
