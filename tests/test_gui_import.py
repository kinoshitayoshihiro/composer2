import importlib

import pytest

streamlit = pytest.importorskip("streamlit")

@pytest.mark.gui
def test_gui_import() -> None:
    assert importlib.import_module("streamlit_app_v2")
