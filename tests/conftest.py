import pytest
from pathlib import Path
from utilities.rhythm_library_loader import load_rhythm_library


def pytest_addoption(parser):
    parser.addoption("--dry-run", action="store_true", help="Run tests in dry-run mode")


@pytest.fixture(scope="session")
def rhythm_library():
    path = Path(__file__).resolve().parents[1] / "data" / "rhythm_library.yml"
    return load_rhythm_library(path)
