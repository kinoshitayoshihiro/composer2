from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class DurationModelBase(ABC):
    """Minimal interface for duration prediction models."""

    @abstractmethod
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Return predicted quarterLength values for note features."""
        raise NotImplementedError
