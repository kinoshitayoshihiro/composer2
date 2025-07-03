# GuitarGenerator API Reference

The `GuitarGenerator` class creates guitar parts with advanced fingering support.
It inherits from `BasePartGenerator` and exposes parameters to control
fretboard positions and movement costs.

## Constructor Fingering Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `position_lock` | `False` | Restrict fingering around `preferred_position`. |
| `preferred_position` | `0` | Central fret when `position_lock` is true. |
| `open_string_bonus` | `-1` | Negative cost favouring open strings. |
| `string_shift_weight` | `2` | Cost for changing strings between notes. |
| `fret_shift_weight` | `1` | Cost for moving frets between notes. |
| `strict_string_order` | `False` | Warn on mismatched arpeggio patterns and auto-adjust. |

Use these options when instantiating the generator or in part parameters to
produce consistent tablature.

