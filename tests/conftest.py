import sys
import types

from tests import _stubs

_stubs.install()

import pytest
from music21 import instrument


@pytest.fixture(autouse=True)
def stub_optional_deps():
    _stubs.install()
    import importlib.util

    for pkg in ("fastapi", "uvicorn", "websockets", "streamlit"):
        try:
            found = importlib.util.find_spec(pkg)
        except ValueError:
            found = None
        if found is None and pkg not in sys.modules:
            mod = types.ModuleType(pkg)
            mod.__spec__ = importlib.machinery.ModuleSpec(pkg, loader=importlib.machinery.BuiltinImporter)
            sys.modules[pkg] = mod

    if "streamlit" not in sys.modules:
        streamlit_module = types.ModuleType("streamlit")
        streamlit_module.cache_data = lambda func: func
        sys.modules["streamlit"] = streamlit_module


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
