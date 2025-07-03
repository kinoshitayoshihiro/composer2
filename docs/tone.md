# Tone and Dynamics

This project can shape bass tone via MIDI control changes. The `ToneShaper`
selects an amp/cabinet preset depending on playing intensity. The chosen preset
is sent as a CC#31 value at the start of the part.

Key switch notes for articulations can be inserted with
`add_key_switches()` from `utilities.articulation_mapper`.

Velocity humanisation optionally maps note volumes to expression (CC11) and
channel aftertouch (CC74). Enable these with the global settings
`use_expr_cc11` and `use_aftertouch`.

## Using ``ToneShaper``

Measure the average note velocity of a part and feed the value to
``ToneShaper.choose_preset`` together with an intensity label.
The returned preset is converted to a CC#31 event which should be inserted at
the start of the part.

```python
from utilities.tone_shaper import ToneShaper

shaper = ToneShaper()
preset = shaper.choose_preset(avg_velocity=80.0, intensity="medium")
part.extra_cc = shaper.to_cc_events(preset, offset=0.0)
```

## Loudness Normalisation

When rendering audio with ``modcompose render`` pass ``--normalize-lufs`` to
target a specific loudness level. The helper ``normalize_wav`` rewrites the WAV
file in place:

```bash
modcompose render spec.yml --soundfont sf2 --normalize-lufs -14
```
