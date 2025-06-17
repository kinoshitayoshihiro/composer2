"""Registry of drum label mappings to General MIDI drum numbers."""

# Additional hi-hat articulations
HH_EDGE = ("closed_hi_hat_edge", 22)
HH_PEDAL = ("pedal_hi_hat", 44)

# Default General MIDI drum map
GM_DRUM_MAP = {
    "chh": ("closed_hi_hat", 42),
    "hh": ("closed_hi_hat", 42),
    "hat_closed": ("closed_hi_hat", 42),
    "hh_edge": HH_EDGE,
    "ohh": ("open_hi_hat", 46),
    "kick": ("acoustic_bass_drum", 36),
    "snare": ("acoustic_snare", 38),
    "ghost_snare": ("acoustic_snare", 38),
    "tom1": ("high_tom", 48),
    "tom2": ("mid_tom", 47),
    "tom3": ("low_tom", 45),
    "crash": ("crash_cymbal_1", 49),
    "crash_cymbal_soft_swell": ("crash_cymbal_1", 49),
    "ride": ("ride_cymbal_1", 51),
    "ride_cymbal_swell": ("ride_cymbal_1", 51),
    "chimes": ("triangle", 81),
    "shaker_soft": ("shaker", 82),
    "ghost": ("closed_hi_hat", 42),
    "brush_kick": ("acoustic_bass_drum", 36),
    "brush_snare": ("acoustic_snare", 38),
    "snare_brush": ("acoustic_snare", 38),
    "hh_pedal": HH_PEDAL,
}

# Mapping for UJAM Virtual Drummer "LEGEND".  Numbers are the note numbers used
# in the plug-in's instrument range.  These roughly follow GM but are included
# separately so users can switch mappings if needed.
UJAM_LEGEND_MAP = {
    "chh": ("closed_hi_hat", 42),
    "hh": ("closed_hi_hat", 42),
    "hat_closed": ("closed_hi_hat", 42),
    "hh_edge": HH_EDGE,
    "ohh": ("open_hi_hat", 46),
    "kick": ("kick", 36),
    "snare": ("snare", 38),
    "ghost_snare": ("snare", 38),
    "tom1": ("high_tom", 50),
    "tom2": ("mid_tom", 47),
    "tom3": ("low_tom", 45),
    "crash": ("crash", 49),
    "crash_cymbal_soft_swell": ("crash", 49),
    "ride_cymbal_swell": ("ride", 51),
    "chimes": ("triangle", 81),
    "shaker_soft": ("shaker", 82),
    "ghost": ("closed_hi_hat", 42),
    "snare_brush": ("snare", 38),
    "hh_pedal": HH_PEDAL,
}

# Fallback mapping for drum names missing from ``GM_DRUM_MAP``.  Used by
# ``generator.drum_generator`` and various scripts to resolve non-standard
# instrument labels.
MISSING_DRUM_MAP_FALLBACK = {
    "hh": "chh",
    "hat_closed": "chh",
    "ghost": "chh",
    "shaker_soft": "shaker_soft",
    "chimes": "chimes",
    "ride_cymbal_swell": "ride_cymbal_swell",
    "crash_cymbal_soft_swell": "crash_cymbal_soft_swell",
}

# Registry of named drum maps
DRUM_MAPS = {
    "gm": GM_DRUM_MAP,
    "ujam_legend": UJAM_LEGEND_MAP,
}

# Backwards compatibility: existing code expects DRUM_MAP to be defined.
DRUM_MAP = GM_DRUM_MAP

def get_drum_map(name: str) -> dict:
    """Return drum map by name with ``gm`` as fallback."""
    return DRUM_MAPS.get(name, GM_DRUM_MAP)
