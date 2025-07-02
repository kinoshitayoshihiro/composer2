# GUI Editor and Presets

The Streamlit GUI now exposes a simple preset system and supports live MIDI capture.

## Recording MIDI

1. Click **Record** to open a virtual MIDI input.
2. Perform on your controller and hit **Stop** when done.
3. The captured notes are previewed instantly.

## Presets

Presets are stored under `~/.otokotoba/presets/` as YAML files. Use the sidebar to save or load presets, or manage them via the CLI:

```bash
modcompose preset list
modcompose preset export my_preset --out preset.yaml
modcompose preset import preset.yaml
```

Apply a preset when rendering a drum pattern:

```bash
modcompose render spec.yml --preset my_preset
```
