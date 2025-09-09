import importlib

import pytest


@pytest.mark.gui
def test_gui_import() -> None:
    importlib.import_module("streamlit_app_v2")
