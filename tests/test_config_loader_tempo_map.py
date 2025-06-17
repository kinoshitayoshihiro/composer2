import yaml
from pathlib import Path
from utilities.config_loader import load_main_cfg


def test_tempo_map_absolute(tmp_path: Path):
    cfg = {
        "global_settings": {
            "time_signature": "4/4",
            "tempo_bpm": 120,
            "key_tonic": "C",
            "key_mode": "major",
            "tempo_map_path": "data/tempo_map.json",
        },
        "paths": {
            "chordmap_path": "dummy",
            "rhythm_library_path": "dummy",
            "output_dir": "out",
        },
        "part_defaults": {},
    }
    cfg_path = tmp_path / "main_cfg.yml"
    cfg_path.write_text(yaml.dump(cfg))
    loaded = load_main_cfg(cfg_path)
    tempo_map = loaded["global_settings"]["tempo_map_path"]
    assert Path(tempo_map).is_absolute()

