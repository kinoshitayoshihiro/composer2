"""Compatibility shim so ``import control_config`` continues to work."""

from utilities.control_config import control_config  # noqa: F401

__all__ = ["control_config"]

