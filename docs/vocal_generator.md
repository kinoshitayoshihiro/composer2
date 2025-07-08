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

## TTS Integration
`scripts/synthesize_vocal.py` reads a MIDI file and a JSON list of
phonemes then calls `tts_model.synthesize` to produce a WAV file.
Invoke it as follows:

```bash
python scripts/synthesize_vocal.py --mid vocal.mid --phonemes phonemes.json --out audio/
```

To use a custom phoneme mapping when sampling from the CLI:

```bash
modcompose sample --backend vocal --phoneme-dict custom_dict.json
```

The output file is written under the directory specified by `--out`.
