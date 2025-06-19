import importlib.util
import warnings
from pathlib import Path

import pytest
from utilities.rhythm_library_loader import load_rhythm_library


REQUIRED_PACKAGES = ["music21", "pretty_midi", "mido"]


def pytest_configure(config):
    missing = [pkg for pkg in REQUIRED_PACKAGES if importlib.util.find_spec(pkg) is None]
    if missing:
        warnings.warn(
            "Missing packages: {}. Install them with 'bash setup.sh' or 'pip install -r requirements.txt'.".format(
                ", ".join(missing)
            )
        )


def pytest_addoption(parser):
    parser.addoption("--dry-run", action="store_true", help="Run tests in dry-run mode")


@pytest.fixture(scope="session")
def rhythm_library():
    path = Path(__file__).resolve().parents[1] / "data" / "rhythm_library.yml"
    return load_rhythm_library(path)
