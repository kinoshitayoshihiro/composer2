import yaml
from pathlib import Path
from utilities.config_loader import load_main_cfg


def test_load_cfg_paths_list(tmp_path: Path):
    cfg = {"paths": {"dummy_files": ["a.txt", "b.txt"]}}
    cfg_path = tmp_path / "cfg.yml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    loaded = load_main_cfg(cfg_path, strict=False)
    dummy = loaded["paths"]["dummy_files"]
    assert isinstance(dummy, list)
    assert all(isinstance(p, str) for p in dummy)

