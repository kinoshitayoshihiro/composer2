from pathlib import Path

import yaml

from utilities.loader import load_chordmap


def test_load_chordmap_fields(tmp_path: Path) -> None:
    data = {
        "global_settings": {},
        "sections": {
            "Intro": {
                "order": 1,
                "length_in_measures": 2,
                "vocal_midi_path": "intro.mid",
                "consonant_json": "intro.json",
            }
        },
    }
    path = tmp_path / "chordmap.yaml"
    path.write_text(yaml.safe_dump(data))
    loaded = load_chordmap(path)
    sec = loaded["sections"]["Intro"]
    assert sec["vocal_midi_path"].endswith("intro.mid")
    assert sec["consonant_json"].endswith("intro.json")


def test_load_chordmap_defaults(tmp_path: Path) -> None:
    data = {
        "global_settings": {"sections": {"A": {"order": 1, "length_in_measures": 1}}}
    }
    path = tmp_path / "chordmap.yaml"
    path.write_text(yaml.safe_dump(data))
    loaded = load_chordmap(path)
    sec = loaded["sections"]["A"]
    assert sec["vocal_midi_path"] == str((tmp_path / "A_vocal.mid").resolve())
    assert sec["consonant_json"] is None


def test_load_chordmap_no_global_settings(tmp_path: Path) -> None:
    data = {"sections": {"B": {"order": 1, "length_in_measures": 1}}}
    path = tmp_path / "chordmap.yaml"
    path.write_text(yaml.safe_dump(data))
    loaded = load_chordmap(path)
    sec = loaded["sections"]["B"]
    assert sec["vocal_midi_path"].endswith("B_vocal.mid")
    assert sec["consonant_json"] is None
