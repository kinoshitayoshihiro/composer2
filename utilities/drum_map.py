from __future__ import annotations
from typing import Dict

# Basic General MIDI drum mapping
GENERAL_MIDI_MAP: Dict[str, int] = {
    "kick": 36,
    "bd": 36,
    "acoustic_bass_drum": 35,
    "snare": 38,
    "sd": 38,
    "acoustic_snare": 38,
    "electric_snare": 40,
    "closed_hi_hat": 42,
    "chh": 42,
    "closed_hat": 42,
    "pedal_hi_hat": 44,
    "phh": 44,
    "open_hi_hat": 46,
    "ohh": 46,
    "open_hat": 46,
    "crash_cymbal_1": 49,
    "crash": 49,
    "crash_cymbal_2": 57,
    "crash_cymbal_soft_swell": 49,
    "ride_cymbal_1": 51,
    "ride": 51,
    "ride_cymbal_2": 59,
    "ride_bell": 53,
    "hand_clap": 39,
    "claps": 39,
    "side_stick": 37,
    "rim": 37,
    "rim_shot": 37,
    "low_floor_tom": 41,
    "tom_floor_low": 41,
    "high_floor_tom": 43,
    "tom_floor_high": 43,
    "low_tom": 45,
    "tom_low": 45,
    "low_mid_tom": 47,
    "tom_mid_low": 47,
    "tom_mid": 47,
    "high_mid_tom": 48,
    "tom_mid_high": 48,
    "tom1": 48,
    "high_tom": 50,
    "tom_hi": 50,
    "tom2": 47,
    "tom3": 45,
    "hat": 42,
    "stick": 31,
    "tambourine": 54,
    "splash": 55,
    "splash_cymbal": 55,
    "cowbell": 56,
    "china": 52,
    "china_cymbal": 52,
    "shaker": 82,
    "cabasa": 69,
    "triangle": 81,
    "wood_block_high": 76,
    "high_wood_block": 76,
    "wood_block_low": 77,
    "low_wood_block": 77,
    "guiro_short": 73,
    "short_guiro": 73,
    "guiro_long": 74,
    "long_guiro": 74,
    "claves": 75,
    "bongo_high": 60,
    "high_bongo": 60,
    "bongo_low": 61,
    "low_bongo": 61,
    "conga_open": 62,
    "mute_high_conga": 62,
    "conga_slap": 63,
    "open_high_conga": 63,
    "timbale_high": 65,
    "high_timbale": 65,
    "timbale_low": 66,
    "low_timbale": 66,
    "agogo_high": 67,
    "high_agogo": 67,
    "agogo_low": 68,
    "low_agogo": 68,
    "ghost_snare": 38,
}

_DRUM_MAPS = {
    "gm": GENERAL_MIDI_MAP,
    "general_midi": GENERAL_MIDI_MAP,
    "default": GENERAL_MIDI_MAP,
}

def get_drum_map(name: str | None) -> Dict[str, int]:
    """Return a drum pitch mapping by name.

    Parameters
    ----------
    name : str | None
        Name of the mapping. ``None`` or unknown names return the General MIDI
        map.
    """
    if not name:
        return GENERAL_MIDI_MAP
    return _DRUM_MAPS.get(name.lower(), GENERAL_MIDI_MAP)

__all__ = ["GENERAL_MIDI_MAP", "get_drum_map"]
