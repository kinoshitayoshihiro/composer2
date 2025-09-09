from __future__ import annotations

import numpy as np


class KDEVelocityModel:
    """Simple KDE-based velocity predictor.

    This minimal implementation returns a constant velocity until fitted.
    """

    def __init__(self, values: np.ndarray | None = None) -> None:
        if values is None:
            values = np.array([64.0], dtype=np.float32)
        self.values = values.astype(np.float32)

    def fit(self, values: np.ndarray) -> None:
        self.values = values.astype(np.float32)

    def predict(self, ctx: np.ndarray, *, cache_key: str | None = None) -> np.ndarray:
        mean = float(np.mean(self.values))
        return np.full((ctx.shape[0],), mean, dtype=np.float32)


__all__ = ["KDEVelocityModel"]
