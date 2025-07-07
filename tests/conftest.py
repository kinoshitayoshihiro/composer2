import importlib.util
import warnings
from pathlib import Path

import pytest

from utilities.rhythm_library_loader import load_rhythm_library

# Ensure sitecustomize stubs are loaded for environments missing optional tools

REQUIRED_PACKAGES = ["music21", "pretty_midi", "mido"]
missing = [pkg for pkg in REQUIRED_PACKAGES if importlib.util.find_spec(pkg) is None]
if missing:
    pytest.skip(
        "Missing packages: {}".format(", ".join(missing)),
        allow_module_level=True,
    )


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


def _opt_dep_available(mod: str) -> bool:
    """Return True if optional dependency *mod* can be imported."""
    return importlib.util.find_spec(mod) is not None


opt_pkgs_missing = any(
    importlib.util.find_spec(m) is None for m in ("soundfile", "scipy", "sklearn")
)


@pytest.fixture(scope="session")
def rhythm_library():
    path = Path(__file__).resolve().parents[1] / "data" / "rhythm_library.yml"
    return load_rhythm_library(path)


def _midi_port_available() -> bool:
    mido = pytest.importorskip("mido", reason="mido not installed")
    try:
        return bool(mido.get_input_names())
    except Exception:
        return False


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item: pytest.Item) -> None:
    if "requires_audio" in item.keywords and opt_pkgs_missing:
        pytest.skip("optional audio deps not installed")
    if "no_midi_port" in item.keywords and not _midi_port_available():
        pytest.skip("no midi port")


@pytest.fixture
def _basic_gen():
    from music21 import instrument

    from generator.guitar_generator import GuitarGenerator

    def factory(**kwargs):
        return GuitarGenerator(
            global_settings={},
            default_instrument=instrument.Guitar(),
            part_name="g",
            global_tempo=120,
            global_time_signature="4/4",
            global_key_signature_tonic="C",
            global_key_signature_mode="major",
            **kwargs,
        )

    return factory
