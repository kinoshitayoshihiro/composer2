from pathlib import Path

import yaml

from utilities.emotion_arranger import generate_bass_arrangement


def test_generate_bass_arrangement(tmp_path: Path) -> None:
    chordmap = {
        "global_settings": {
            "tempo": 100,
            "time_signature": "4/4",
            "key_tonic": "C",
            "key_mode": "major",
        },
        "sections": {
            "Intro": {
                "order": 1,
                "length_in_measures": 2,
                "musical_intent": {
                    "emotion": "quiet_pain_and_nascent_strength",
                    "intensity": "low",
                },
            }
        },
    }
    chordmap_path = tmp_path / "chordmap.yaml"
    chordmap_path.write_text(yaml.safe_dump(chordmap))

    arrangement = generate_bass_arrangement(
        chordmap_path,
        Path("data/rhythm_library.yml"),
        Path("data/emotion_profile.yaml"),
    )

    assert arrangement["Intro"]["bass_pattern_key"] == "root_only"
