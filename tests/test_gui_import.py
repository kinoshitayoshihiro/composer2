import importlib

import pytest

streamlit = pytest.importorskip("streamlit")

def test_gui_import() -> None:
    assert importlib.import_module("streamlit_app_v2")
