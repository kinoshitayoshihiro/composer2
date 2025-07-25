import sys
import types

import pytest
from music21 import instrument


@pytest.fixture(autouse=True)
def stub_optional_deps():
    # FastAPI stub with more methods
    class FastAPIStub:
        def __init__(self):
            pass

        def post(self, path):
            return lambda f: f

        def middleware(self, name):
            def decorator(f):
                return f

            return decorator

    class ResponseStub:
        def __init__(self, content, status_code=200):
            self.content = content
            self.status_code = status_code

    fastapi_module = types.ModuleType("fastapi")
    fastapi_module.FastAPI = FastAPIStub
    fastapi_module.Request = type("Request", (), {})
    fastapi_module.HTTPException = type("HTTPException", (Exception,), {})

    # FastAPI responses submodule
    responses_module = types.ModuleType("fastapi.responses")
    responses_module.JSONResponse = ResponseStub
    sys.modules["fastapi"] = fastapi_module
    sys.modules["fastapi.responses"] = responses_module

    # Other stubs
    for pkg in ("uvicorn", "websockets"):
        sys.modules[pkg] = types.ModuleType(pkg)

    # Streamlit stub with cache_data
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
