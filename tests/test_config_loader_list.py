import yaml
from pathlib import Path

from utilities.config_loader import load_main_cfg


def test_list_path_handled(tmp_path: Path) -> None:
    """drum_pattern_files がリストでも正しくロードできるか"""
    cfg = {
        "global_settings": {
            "time_signature": "4/4",
            "tempo_bpm": 120,
            "key_tonic": "C",
            "key_mode": "major",
        },
        "paths": {
            "chordmap_path": "dummy",
            "rhythm_library_path": "dummy",
            "output_dir": "out",
            "drum_pattern_files": ["data/a.yml", "data/b.yml"],
        },
        "part_defaults": {},
    }

    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    loaded = load_main_cfg(cfg_path)
    files = loaded["paths"]["drum_pattern_files"]

    assert isinstance(files, list)
    assert all(isinstance(f, str) for f in files)


def test_load_cfg_paths_list(tmp_path: Path) -> None:
    """paths.* に任意キーを置いた場合でもリストとして読み込めるか"""
    cfg = {"paths": {"dummy_files": ["a.txt", "b.txt"]}}

    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    loaded = load_main_cfg(cfg_path, strict=False)
    dummy = loaded["paths"]["dummy_files"]

    assert isinstance(dummy, list)
    assert all(isinstance(p, str) for p in dummy)
