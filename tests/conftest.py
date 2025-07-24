import sys
import types

import pytest


@pytest.fixture(autouse=True)
def stub_optional_deps():
    for pkg in ("fastapi", "uvicorn", "websockets", "streamlit"):
        sys.modules[pkg] = types.ModuleType(pkg)
