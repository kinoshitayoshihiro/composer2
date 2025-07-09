# Vocal Generator

## Lyrics & Phonemes
The `VocalGenerator.compose` method accepts a `lyrics_words` parameter.
When present the lyrics string is split into syllables and converted to
phonemes via `text_to_phonemes`.  The mapping dictionary is applied
in greedy order so multi-character keys such as "きゃ" resolve correctly.

## Vibrato
`generate_vibrato(duration_qL, depth, rate, step=0.05)` returns a list
of pitch-wheel and aftertouch events.  Depth is specified in semitones
and rate in cycles per quarter note.
The events are attached to each note in
`_apply_vibrato_to_part`.


```python
from utilities.vibrato_engine import generate_vibrato

# example: 1 qL note with 0.5 semitone depth at 5 Hz
events = generate_vibrato(1.0, 0.5, 5.0)
```

## Expression & Vibrato

`VocalGenerator` exposes articulation parameters so that vibrato and special
markers can be toggled from the CLI or config file.  These options are passed to
the helper functions `generate_vibrato`, `generate_gliss` and
`generate_trill`:

* `--vibrato-depth` – vibrato depth in semitones (default `0.5`)
* `--vibrato-rate` – oscillation rate in cycles per quarter note (default `5.0`)
* `--enable-articulation` / `--no-enable-articulation` – toggle vibrato,
  glissando and trill generation (enabled by default)

The same keys may be placed under `part_defaults.vocal` in `main_cfg.yml` to
avoid long command-line flags.

## TTS Integration
`scripts/synthesize_vocal.py` reads a MIDI file and a JSON list of
phonemes then calls `tts_model.synthesize` to produce a WAV file.
Invoke it as follows:

```bash
python scripts/synthesize_vocal.py --mid vocal.mid --phonemes phonemes.json --out audio/
```

## TTS ONNX Integration
To run synthesis through an ONNX model pass the `--onnx-model` option and specify a log level:

```bash
python scripts/synthesize_vocal.py --mid vocal.mid --phonemes phonemes.json \
    --out audio/ --onnx-model model.onnx --log-level DEBUG
```

Exit status is `0` when the WAV is created successfully and `1` if synthesis fails.

To use a custom phoneme mapping when sampling from the CLI:

```bash
modcompose sample --backend vocal --phoneme-dict custom_dict.json
```

Adjust vibrato behaviour with `--vibrato-depth` and `--vibrato-rate`:

```bash
modcompose sample model.pt --backend vocal \
    --vibrato-depth 0.6 --vibrato-rate 6
```

The output file is written under the directory specified by `--out`.
