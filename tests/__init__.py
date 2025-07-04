import importlib
import pytest


def skip_if_no_torch(allow_module_level: bool = False) -> None:
    """Skip test if PyTorch is unavailable."""
    try:
        importlib.import_module("torch")
    except Exception:
        pytest.skip("PyTorch not installed", allow_module_level=allow_module_level)
