# ToneShaper Reference

`ToneShaper` selects an amp/cabinet preset based on playing intensity and average note velocity. Presets are stored in `data/preset_library.yml` and can be overridden by calling `ToneShaper.load_presets()` or by setting the environment variable `PRESET_LIBRARY_PATH`.
If the variable is set it takes priority over the default path. Calling `load_presets(None)` will reload from `PRESET_LIBRARY_PATH` when present.

## Preset Library の検索優先順位

```
Environment variable > load_presets(path) > default_path
```

## Loading and Reloading

```python
from utilities.tone_shaper import ToneShaper

# load custom preset library
ToneShaper.load_presets("my_presets.yml")
shaper = ToneShaper()

# update library at runtime
shaper.reload_presets()
```

`reload_presets()` re-reads the library file and updates the internal mappings. Existing `ToneShaper` instances adopt the new presets while keeping the current selection whenever possible.

`load_presets()` validates that all CC values fall within the MIDI range 0-127. Invalid values raise `ValueError` at load time.
When an IR file listed in the library does not exist the entry is kept but its
`ir_file` value becomes `None` and a warning is logged.

## Style Hint

`choose_preset()` accepts a `style` argument which is treated the same as `amp_hint`. If the style is unknown, or an unrecognised intensity is passed, the method safely returns `default_preset`.

```
shaper.choose_preset(style="rock", intensity="high", avg_velocity=80)
```

## Environment Variables

- `PRESET_LIBRARY_PATH` – path to a preset library file loaded implicitly on import and when calling `load_presets()` with no argument.

`get_ir_file(preset_name, fallback_ok=False)` now checks that the preset's IR file actually exists. When the file is missing a `FileNotFoundError` is raised unless `fallback_ok=True` is given, in which case the method logs a warning and falls back to the `clean` preset's IR if available.

`render_with_ir()` fetches `gain_db` and the loudness target (`lufs` or `gain_db` field) from the library when not explicitly provided. The default loudness target is -14 LUFS.

Use `export_mix_json` to store mix metadata. When the selected preset has an IR file and the part metadata lacks one, the IR path from `ToneShaper.get_ir_file()` is inserted automatically.

## Optional Dependencies

Audio rendering utilities rely on extra packages. Install them via:

```bash
pip install modular-composer[audio,ml]
```

This provides `soundfile`, `scipy`, and `scikit-learn` used by ToneShaper tests and mix helpers.
