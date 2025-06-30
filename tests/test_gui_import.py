import pytest

pytestmark = pytest.mark.stretch


def test_gui_import():
    import streamlit_app.gui  # noqa: F401
