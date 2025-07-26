import sys
import types
import importlib.util
import importlib.machinery
from importlib.util import spec_from_loader

# --- kill broken fastapi stub ASAP (before anything else imports it) ---
if "fastapi" in sys.modules and getattr(sys.modules["fastapi"], "__spec__", None) is None:
    del sys.modules["fastapi"]

# Import stubs for optional dependencies early so music21 import works
try:
    from . import _stubs
except ImportError:
    _stubs = types.ModuleType("_stubs")
    _stubs.install = lambda: None

_stubs.install()

import pytest
from music21 import instrument


@pytest.fixture(autouse=True)
def stub_optional_deps():
    """Install lightweight stubs for truly optional packages."""
    _stubs.install()
    if importlib.util.find_spec("fastapi") is None:
        sys.modules.pop("fastapi", None)

    def _stub_pkg(name: str, attrs: dict | None = None) -> None:
        # leave real packages untouched
        try:
            if importlib.util.find_spec(name):
                return
        except ValueError:
            pass
        if name in sys.modules and getattr(sys.modules[name], "__spec__", None):
            return

        mod = sys.modules.get(name, types.ModuleType(name))
        spec = spec_from_loader(name, loader=None, is_package=True)
        spec.submodule_search_locations = []
        mod.__spec__ = spec
        mod.__package__ = name
        mod.__path__ = []
        if attrs:
            for k, v in attrs.items():
                setattr(mod, k, v)
        sys.modules[name] = mod

    for pkg in ("uvicorn", "websockets", "streamlit"):
        _stub_pkg(pkg)

    st = sys.modules.get("streamlit")
    if st and not hasattr(st, "cache_data"):
        st.cache_data = lambda func: func


@pytest.fixture
def _basic_gen():
    """Basic GuitarGenerator fixture for testing."""
    from generator.guitar_generator import GuitarGenerator

    def _create_generator(**kwargs):
        # デフォルト設定
        default_args = {
            "global_settings": {},
            "default_instrument": instrument.Guitar(),
            "part_name": "g",
            "global_tempo": 120,
            "global_time_signature": "4/4",
            "global_key_signature_tonic": "C",
            "global_key_signature_mode": "major",
        }
        # kwargsでデフォルト設定を上書き
        default_args.update(kwargs)

        return GuitarGenerator(**default_args)

    return _create_generator
