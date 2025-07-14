"""Common type aliases used across utilities."""
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:  # pragma: no cover - optional imports for typing
    import plotly.graph_objects as go
    import sounddevice
    import streamlit
    import torch

Intensity = Literal["low", "mid", "high"]
AuxTuple = tuple[str, str, str]

__all__ = ["Intensity", "AuxTuple"]

