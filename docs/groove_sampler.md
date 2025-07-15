# Groove Sampler

Train an n-gram model from a folder of MIDI or WAV loops.

Install optional dependencies to enable WAV extraction and the CLI:

```bash
pip install click
pip install -e .[audio]
```

```bash
modcompose groove train loops/ --ext midi --out model.pkl
```

Generate a four bar MIDI groove:

```bash
modcompose groove sample model.pkl -l 4 --temperature 0.8 --seed 42 > out.mid
```

### New parameters

``groove_sampler_v2`` exposes additional options:

- ``--beats-per-bar``: override bar length when inferring resolution
- ``--temperature-end``: final sampling temperature for scheduling
- ``--top-k`` / ``--top-p``: filter sampling candidates

Example:

```bash
groove_sampler_v2 train loops/ --auto-res --beats-per-bar 8
groove_sampler_v2 sample model.pkl -l 4 --top-k 5 --top-p 0.8 > out.json
```

An experimental RNN baseline can be trained from the cached loops:

```bash
modcompose rnn train loops.json --epochs 6 --out rnn.pt
modcompose rnn sample rnn.pt -l 4 > rnn.mid
```

## Aux-feature training / sampling

You can condition groove generation on section type, vocal heatmap bin and
intensity bucket. Provide a JSON file mapping each loop filename to auxiliary
data when training and use `--cond` at sampling time:

```bash
modcompose groove train loops/ --aux aux.json
modcompose groove sample model.pkl --cond '{"section":"chorus","intensity":"high"}' > out.mid
```

### Backward compatibility

If you omit the `--aux` option during training, the sampler behaves exactly as
in version 1.0 and ignores auxiliary conditions.

Models generated prior to commit 608fdda no longer include the
deprecated `aux_dims` field and should be retrained.

## Style/Aux Tagging

Loops may include a sidecar YAML file with optional tags:

```yaml
style: lofi
feel: relaxed
density: sparse
source: freesound
```

Train with one of these tags using `--aux-key` and condition at sample time:

```bash
python -m utilities.groove_sampler_v2 train loops/ -o model.pkl --aux-key style
python -m utilities.groove_sampler_v2 sample model.pkl --cond-style lofi -l 4
```

In YAML configs you can set:

```yaml
parts:
  drums:
    cond: {style: funk}
```

## CLI Commands

| Command | Description | Key options |
| ------- | ----------- | ----------- |
| `groove train` | Train n-gram model from loops | `--ext`, `--out` |
| `groove sample` | Generate groove MIDI | `-l`, `--temperature`, `--seed` |
| `export-midi` | Save sampled MIDI to file | `--length`, `--temperature`, `--seed` |
| `render-audio` | Convert MIDI to audio with FluidSynth | `--out`, `--soundfont`, `--use-default-sf2` |
| `evaluate` | Calculate basic groove metrics | `--ref` |
| `visualize` | Draw n-gram frequency heatmap | `--out` |
| `hyperopt` | Optuna search over temperature | `--trials`, `--skip-if-no-optuna` |

## Audio → MIDI Batch

Convert drum loops into MIDI files using BasicPitch:

```bash
python -m utilities.audio_to_midi_batch loops/wav midi_out --jobs 4
```

The converter runs on CPU and works best with 44.1 kHz audio. Cache
`~/.cache/basic_pitch` to speed up repeated runs.

### Audio → MIDI Batch CLI

Use the batch converter to transcribe a folder of loops:

```bash
python -m utilities.audio_to_midi_batch loops/wav midi_out \
    --part drums --ext wav,mp3 --jobs 4 --min-db -35
```

Pass `--stem` to restrict processing to a specific filename and add
`--overwrite` to regenerate existing MIDI files.

Requires the `libsndfile` system library for WAV/FLAC support.
