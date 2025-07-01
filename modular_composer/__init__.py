__all__ = ["__version__"]

__version__ = "1.0.1"

from importlib import util
from pathlib import Path

_spec = util.spec_from_file_location(
    "_legacy_modular_composer", Path(__file__).resolve().parent.parent / "modular_composer.py"
)
if _spec and _spec.loader:
    _legacy = util.module_from_spec(_spec)
    _spec.loader.exec_module(_legacy)
    for name in dir(_legacy):
        if not name.startswith("_"):
            globals()[name] = getattr(_legacy, name)
