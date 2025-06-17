import pytest

from utilities.drum_map_registry import UJAM_LEGEND_MAP

EXPECTED = {
    "chh": 42,
    "hh": 42,
    "hat_closed": 42,
    "ohh": 46,
    "kick": 36,
    "snare": 38,
    "ghost_snare": 38,
    "tom1": 50,
    "tom2": 47,
    "tom3": 45,
    "crash": 49,
    "crash_cymbal_soft_swell": 49,
    "ride_cymbal_swell": 51,
    "chimes": 81,
    "shaker_soft": 82,
    "ghost": 42,
<<<<<<< codex/update-drum-map-and-brush_mode-functionality
    "snare_brush": 38,
=======
    "hh_edge": 44,
    "hh_pedal": 44,
>>>>>>> infra/zero-green
}

def test_ujam_legend_map_notes():
    for name, (instrument, midi) in UJAM_LEGEND_MAP.items():
        assert name in EXPECTED
        assert midi == EXPECTED[name], f"{name} -> {midi}, expected {EXPECTED[name]}"
