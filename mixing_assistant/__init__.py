"""Mixing assistant package."""

from importlib import import_module


def download_refs(*args, **kwargs):
    """Lazy wrapper for :func:`download_ref_masters.download_refs`."""
    mod = import_module("mixing_assistant.download_ref_masters")
    return mod.download_refs(*args, **kwargs)


def extract_features(*args, **kwargs):
    """Lazy wrapper for :func:`feature_extractor.extract_features`."""
    mod = import_module("mixing_assistant.feature_extractor")
    return mod.extract_features(*args, **kwargs)

__all__ = ["download_refs", "extract_features"]
