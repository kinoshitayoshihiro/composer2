import sys
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
pkg = type(sys)("generator")
pkg.__path__ = [str(ROOT / "generator")]
sys.modules.setdefault("generator", pkg)

_MOD_PATH = ROOT / "generator" / "strings_generator.py"
spec = importlib.util.spec_from_file_location("generator.strings_generator", _MOD_PATH)
strings_module = importlib.util.module_from_spec(spec)
sys.modules["generator.strings_generator"] = strings_module
spec.loader.exec_module(strings_module)

parse_bow_position = strings_module.parse_bow_position
BowPosition = strings_module.BowPosition


def test_bow_position_aliases() -> None:
    assert parse_bow_position("sul pont.") == BowPosition.PONTICELLO
    assert parse_bow_position("sul tasto") == BowPosition.TASTO


def test_bow_position_invalid() -> None:
    assert parse_bow_position("mystery") is None

