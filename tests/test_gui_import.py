import pytest

pytestmark = pytest.mark.stretch


def test_gui_import():
    try:
        import streamlit_app.gui  # noqa: F401
    except ModuleNotFoundError:
        pytest.skip("streamlit not installed")
