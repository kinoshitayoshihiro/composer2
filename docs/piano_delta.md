# Piano Delta Features

## Tone Presets

Select from built-in tone presets when rendering demo tracks:

- `grand_clean` – balanced grand piano with subtle EQ
- `upright_mellow` – softer timbre with low-pass filter
- `ep_phase` – DX-style electric piano with chorus

Use `--tone-preset` with `modcompose sample` to choose one of these.

## Articulation Tags

Chord symbols may contain the words `gliss` or `trill` to trigger short glissando
or trill passages. Disable this behaviour with `--no-enable-articulation`.

## Loudness Workflow

`normalize_velocities()` scales MIDI velocities to match a target LUFS level.
Rendered WAV files can be normalised per section using `normalize_wav()`.
