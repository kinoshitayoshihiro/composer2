"""Default control configuration shim used by generators.

This module intentionally keeps dependencies light so that tests can import
``utilities.control_config`` even in environments where optional packages are
missing.  The default ``control_config`` mirrors the legacy behaviour by
providing a handful of commonly accessed attributes.
"""

from __future__ import annotations

from types import SimpleNamespace

control_config = SimpleNamespace(
    swing_ratio=0.0,
    jitter_ms=0.0,
    velocity_curve="linear",
    enable_cc11=True,
    enable_cc64=False,
)

__all__ = ["control_config"]

