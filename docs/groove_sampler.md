# Groove Sampler

Train an n-gram model from a folder of MIDI or WAV loops.

![WebSocket Bridge](https://example.com/ws_bridge.gif)

## Table of Contents
- [New parameters](#new-parameters)
- [Aux-feature training / sampling](#aux-feature-training--sampling)
- [Style/Aux Tagging](#styleaux-tagging)
- [CLI Commands](#cli-commands)
- [Audio → MIDI Batch](#audio--midi-batch)
- [Percussion workflow](#percussion-workflow)

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
- ``--tempo-policy``: how to handle missing/invalid tempo (skip|fallback|accept)
- ``--fallback-bpm``: BPM used when ``tempo-policy=fallback`` (default 120)
- ``--min-bpm`` / ``--max-bpm``: flag tempos outside this range as invalid
- ``--fold-halves``: fold near double/half-time tempos into the valid range
- ``--tempo-verbose``: print summary statistics for tempo handling
- ``--min-bars`` / ``--min-notes``: skip loops shorter than a bar or lacking notes
- ``--drum-only`` / ``--pitched-only``: restrict to drum or melodic material
- ``--exclude-fills``: drop files tagged as fills (by filename)
- ``--len-sampling``: weighting for loop length (uniform|sqrt|proportional)
- ``--inject-default-tempo``: write a tempo event when missing (0 disables)

A separate `scripts/scan_loops.py` utility inventories a loop folder and
produces `loop_inventory.csv` with per-file statistics:

```bash
python scripts/scan_loops.py --root data/loops --out loop_inventory.csv
```

Example:

```bash
groove_sampler_v2 train loops/ --auto-res --beats-per-bar 8
groove_sampler_v2 sample model.pkl -l 4 --top-k 5 --top-p 0.8 > out.json
```

## Memory & Scale

Large loop corpora can exceed RAM when every n-gram count is stored in memory.
The trainer supports spilling intermediate counts to disk using ``numpy``
memmaps.

- ``--memmap-dir``: directory for memmap shards
- ``--snapshot-interval``: flush counts every N files
- ``--counts-dtype``: dtype for on-disk counters (``uint32`` or ``uint16``)

Example workflow:

```bash
python -m utilities.groove_sampler_v2 train loops/ \
    --memmap-dir tmp/mm \
    --snapshot-interval 10 \
    --counts-dtype uint16 \
    -o model.pkl
```

Each snapshot updates the memmap arrays and frees the temporary buffers so that
peak RSS remains low.  After training completes the shards are merged into the
final model and the temporary directory can be removed.

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

## Percussion workflow

Train a simple n‑gram model from percussion loops and sample bars:

```bash
python -m utilities.perc_sampler_v1 train data/loops/percussion -o models/perc_ngram.pkl --auto-res
python -m utilities.perc_sampler_v1 sample models/perc_ngram.pkl --length 4 > perc.mid
```

Add a percussion part to your configuration:

```yaml
parts:
  percussion:
    backend: perc_ngram
    model_path: models/perc_ngram.pkl
```

Percussion hits colliding with a kick or snare are shifted forward by one tick when merging with the drum track.

## Advanced ML (Transformer)

An experimental Transformer model can jointly generate drums, bass, piano and percussion. It uses a `TransformerEncoder` implemented with PyTorch Lightning.

```
drums  ┐
bass   ┤  Embedding + PosEnc → Transformer → separate heads → predictions
piano  ┤
perc   ┘
```

Train on token text files for each part:

```bash
python tools/train_transformer.py data/loops --parts drums,bass --epochs 1 --batch 8
```

Sample a short phrase:

```bash
python -m utilities.transformer_sampler sample models/groove_transformer.ckpt --parts drums,bass --length 8
```

## Vocal Synthesis Integration

Install system library and extras:

```bash
sudo apt-get install -y libsndfile1
poetry install .[audio]
```

Train a vocal model:

```bash
python -m tools.train_vocal_model \
  --lyrics "Hello world" \
  --model-output models/vocal.ckpt
```

Synthesize vocals from the model:

```bash
python utilities/vocal_synth.py \
  --model models/vocal.ckpt \
  --output out.wav
```

Run as a FastAPI server:

```python
from fastapi import FastAPI, Response

from utilities.vocal_synth import synthesize

app = FastAPI()

@app.post("/vocal")
async def vocal_endpoint():
    audio = synthesize("models/vocal.ckpt")
    return Response(content=audio, media_type="audio/wav")

```

Start the server with:

```bash
uvicorn my_server:app --reload
```

## Phase 3: Data Augmentation

Use the new augmentation CLI to expand the training data.

Run augmentation and rebuild the velocity CSV:

```bash
python scripts/train_velocity.py augment-data \
  --wav-dir data/tracks \
  --out-dir data/tracks_aug \
  --drums-dir data/loops/drums \
  --shifts -2,0,2 \
  --rates 0.8,1.2 \
  --snrs 20,10
```

- `--shifts`: pitch shift amounts in semitones
- `--rates`: time stretch rates
- `--snrs`: signal-to-noise ratios for added noise
- `--drums-dir`: directory containing drum loops for CSV rebuild

Training with `--augment` applies random transforms on the fly.
