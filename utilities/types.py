"""Common type aliases used across utilities."""
from typing import Literal

Intensity = Literal["low", "mid", "high"]
AuxTuple = tuple[str, str, str]

__all__ = ["Intensity", "AuxTuple"]

