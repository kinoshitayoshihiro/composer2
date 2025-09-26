__all__ = ["__version__"]

__version__ = "3.0.0"

from importlib import util
from pathlib import Path

_spec = util.spec_from_file_location(
    "_legacy_modular_composer",
    Path(__file__).resolve().parent.parent / "modular_composer.py",
)
if _spec and _spec.loader:
    _legacy = util.module_from_spec(_spec)
    try:
        _spec.loader.exec_module(_legacy)
    except ModuleNotFoundError:
        _legacy = None
    if _legacy is not None:
        for name in dir(_legacy):
            if not name.startswith("_"):
                globals()[name] = getattr(_legacy, name)
