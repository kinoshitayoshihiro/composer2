from __future__ import annotations

import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

try:
    import pretty_midi  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    pretty_midi = None  # type: ignore


class KDEVelocityModel:
    """Simple velocity sampler based on beat-position histograms."""

    def __init__(self, data: dict[str, dict[int, list[int]]]):
        self.data = data

    @classmethod
    def train(
        cls,
        directory: str | Path,
        *,
        parts: Sequence[str] | None = None,
        out_path: str | Path = "velocity_model.pkl",
    ) -> str:
        if pretty_midi is None:
            raise RuntimeError("pretty_midi not installed; install extras 'midi'")
        directory = Path(directory)
        data: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
        for midi_path in directory.rglob("*.mid"):
            try:
                pm = pretty_midi.PrettyMIDI(str(midi_path))
            except Exception:
                continue
            tempos, _ = pm.get_tempo_changes()
            tempo = float(tempos[0]) if len(tempos) else 120.0
            spb = 60.0 / tempo
            for inst in pm.instruments:
                name = (inst.name or "").lower()
                if parts and name not in parts:
                    continue
                for n in inst.notes:
                    beat = n.start / spb
                    bin_idx = int((beat % 1.0) * 4)
                    data[name][bin_idx].append(int(n.velocity))
        model = cls({p: {b: v for b, v in bins.items()} for p, bins in data.items()})
        with open(out_path, "wb") as fh:
            pickle.dump(model, fh)
        return str(out_path)

    @staticmethod
    def load(path: str | Path) -> "KDEVelocityModel":
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        if isinstance(obj, KDEVelocityModel):
            return obj
        raise TypeError("Invalid velocity model file")

    def sample(self, part: str, pos_beat: float) -> int:
        bins = self.data.get(part.lower())
        if not bins:
            return 64
        bin_idx = int((pos_beat % 1.0) * 4)
        vals = bins.get(bin_idx)
        if not vals:
            # fallback to overall mean
            vals = [v for arr in bins.values() for v in arr]
        if not vals:
            return 64
        return int(random.choice(vals))


__all__ = ["KDEVelocityModel"]
