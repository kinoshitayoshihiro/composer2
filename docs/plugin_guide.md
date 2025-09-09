# Plugin Guide

Build the plugin and install it into your DAW.

```bash
cmake -B build -DMODC_BUILD_PLUGIN=ON
cmake --build build --config Release
```

Copy the resulting `.vst3` or `.clap` file to your plug‑ins folder.

## Guitar Fingering Parameters

When scripting the guitar generator you can tune fingering behaviour.
Important options include:

- `position_lock` and `preferred_position` to keep notes around a certain fret
- `open_string_bonus` to favour open strings when possible
- `string_shift_weight` and `fret_shift_weight` to penalise hand movement
- `strict_string_order` to enforce exact arpeggio patterns

Configure these in your part parameters or generator constructor for
consistent tablature output.
