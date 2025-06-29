import yaml
from pathlib import Path
from utilities.rhythm_library_loader import load_rhythm_library


def test_guitar_patterns_alias(tmp_path: Path) -> None:
    data = {"guitar_patterns": {"demo": {"pattern": []}}}
    path = tmp_path / "lib.yml"
    path.write_text(yaml.safe_dump(data))
    lib = load_rhythm_library(path)
    assert "demo" in (lib.guitar or {})
    dumped = lib.model_dump(by_alias=True)
    assert "guitar_patterns" in dumped

