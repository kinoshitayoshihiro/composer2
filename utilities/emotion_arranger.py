from __future__ import annotations

from pathlib import Path
from typing import Any

from music21 import instrument

from generator.bass_generator import BassGenerator
from utilities.config_loader import load_chordmap_yaml
from utilities.emotion_profile_loader import load_emotion_profile
from utilities.rhythm_library_loader import load_rhythm_library


def generate_bass_arrangement(
    chordmap_path: str | Path,
    rhythm_library_path: str | Path,
    emotion_profile_path: str | Path,
) -> dict[str, dict[str, Any]]:
    """Return bass pattern choices per section based on emotion.

    Parameters
    ----------
    chordmap_path : str | Path
        YAML chord map describing sections and emotions.
    rhythm_library_path : str | Path
        Path to rhythm library YAML.
    emotion_profile_path : str | Path
        Path to emotion profile YAML for BassGenerator.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Mapping of section name to arrangement data. Each entry contains
        ``"bass_pattern_key"`` as well as optional ``"octave_pref"`` and
        ``"length_beats"`` keys derived from the emotion profile.
    """
    chordmap = load_chordmap_yaml(Path(chordmap_path))
    rhythm_lib = load_rhythm_library(str(rhythm_library_path))
    emotion_profiles = load_emotion_profile(emotion_profile_path)

    global_settings = chordmap.get("global_settings", {})
    tempo = int(global_settings.get("tempo", 120))
    ts = str(global_settings.get("time_signature", "4/4"))
    key_tonic = global_settings.get("key_tonic", "C")
    key_mode = global_settings.get("key_mode", "major")

    gen = BassGenerator(
        part_name="bass",
        default_instrument=instrument.AcousticBass(),
        global_settings={},
        global_tempo=tempo,
        global_time_signature=ts,
        global_key_signature_tonic=key_tonic,
        global_key_signature_mode=key_mode,
        main_cfg={"global_settings": {}},
        emotion_profile_path=str(emotion_profile_path),
        part_parameters=rhythm_lib.bass_patterns or {},
    )

    arrangement: dict[str, dict[str, Any]] = {}
    for name, section in chordmap.get("sections", {}).items():
        intent = section.get("musical_intent", {})
        pattern_key = gen._choose_bass_pattern_key(intent)

        emotion = intent.get("emotion", "default")
        emotion_cfg = emotion_profiles.get(emotion, {}) if isinstance(emotion_profiles, dict) else {}

        arrangement[name] = {
            "bass_pattern_key": pattern_key,
            "octave_pref": emotion_cfg.get("octave_pref"),
            "length_beats": emotion_cfg.get("length_beats"),
        }

    return arrangement
