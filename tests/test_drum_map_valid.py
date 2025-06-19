import pytest

from utilities.drum_map_registry import DRUM_MAPS


def test_drum_map_values_unique_and_in_range():
    for name, mapping in DRUM_MAPS.items():
        midi_to_instrument = {}
        for label, (instrument, midi) in mapping.items():
            assert 0 <= midi <= 127, f"{name} {label} -> {midi} out of range"
            if midi in midi_to_instrument:
                if midi not in {44}:
                    assert (
                        midi_to_instrument[midi] == instrument
                    ), f"{name} MIDI {midi} maps to both {midi_to_instrument[midi]} and {instrument}"
            else:
                midi_to_instrument[midi] = instrument

