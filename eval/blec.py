from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import entropy

Event = Mapping[str, float | int | str]


def _hist(events: Iterable[Event], resolution: int) -> dict[str, np.ndarray]:
    bins: dict[str, np.ndarray] = defaultdict(
        lambda: np.zeros(resolution, dtype=np.float64)
    )
    for ev in events:
        inst = str(ev.get("instrument", ""))
        step = int(round(float(ev.get("offset", 0)) * (resolution / 4))) % resolution
        bins[inst][step] += 1
    for arr in bins.values():
        s = arr.sum()
        if s:
            arr /= s
    return bins


def blec(true: Sequence[Event], pred: Sequence[Event], *, resolution: int = 16) -> float:
    """Return Binned Log-Likelihood Error per Class."""
    h_true = _hist(true, resolution)
    h_pred = _hist(pred, resolution)
    all_inst = set(h_true) | set(h_pred)
    scores = []
    for inst in all_inst:
        p = h_true.get(inst, np.ones(resolution) / resolution)
        q = h_pred.get(inst, np.ones(resolution) / resolution)
        scores.append(entropy(p, q) / np.log(resolution))
    return float(np.mean(scores) if scores else 0.0)
