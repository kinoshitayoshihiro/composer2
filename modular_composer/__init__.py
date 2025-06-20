from importlib import util
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

__all__ = ["__version__"]

try:
    __version__ = version("modular_composer")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"

_spec = util.spec_from_file_location(
    "_legacy_modular_composer", Path(__file__).resolve().parent.parent / "modular_composer.py"
)
if _spec and _spec.loader:
    _legacy = util.module_from_spec(_spec)
    _spec.loader.exec_module(_legacy)
    for name in dir(_legacy):
        if not name.startswith("_"):
            globals()[name] = getattr(_legacy, name)
