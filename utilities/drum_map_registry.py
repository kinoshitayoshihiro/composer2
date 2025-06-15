"""Registry of drum label mappings to General MIDI drum numbers."""

DRUM_MAP = {
    "chh": ("closed_hi_hat", 42),
    "hh": ("closed_hi_hat", 42),
    "ohh": ("open_hi_hat", 46),
    "kick": ("acoustic_bass_drum", 36),
    "snare": ("acoustic_snare", 38),
    "ghost_snare": ("acoustic_snare", 38),
    "tom1": ("high_tom", 48),
    "tom2": ("mid_tom", 47),
    "tom3": ("low_tom", 45),
    "crash": ("crash_cymbal_1", 49),
    "crash_cymbal_soft_swell": ("crash_cymbal_1", 49),
    "ride_cymbal_swell": ("ride_cymbal_1", 51),
    "chimes": ("triangle", 81),
    "shaker_soft": ("shaker", 82),
    "ghost": ("closed_hi_hat", 42),
}
