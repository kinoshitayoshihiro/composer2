import importlib.util
import warnings
from pathlib import Path

# Ensure sitecustomize stubs are loaded for environments missing optional tools
import sitecustomize

import pytest

from utilities.rhythm_library_loader import load_rhythm_library

REQUIRED_PACKAGES = ["music21", "pretty_midi", "mido"]


def pytest_configure(config):
    missing = [pkg for pkg in REQUIRED_PACKAGES if importlib.util.find_spec(pkg) is None]
    if missing:
        msg = (
            "Missing packages: {}. Install them with "
            "'bash setup.sh' or 'pip install -r requirements.txt'."
        )
        warnings.warn(msg.format(", ".join(missing)))


def pytest_addoption(parser):
    parser.addoption("--dry-run", action="store_true", help="Run tests in dry-run mode")
    parser.addoption(
        "--generate-golden",
        action="store_true",
        help="Regenerate golden MIDI files",
    )
    parser.addoption(
        "--update-golden",
        action="store_true",
        help="Regenerate golden MIDI files",
    )


@pytest.fixture(scope="session")
def rhythm_library():
    path = Path(__file__).resolve().parents[1] / "data" / "rhythm_library.yml"
    return load_rhythm_library(path)


def _midi_port_available() -> bool:
    try:
        import mido
        return bool(mido.get_input_names())
    except Exception:
        return False


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item: pytest.Item) -> None:
    if "no_midi_port" in item.keywords and not _midi_port_available():
        pytest.skip("no midi port")
