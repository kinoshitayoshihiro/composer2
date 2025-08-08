# OtoKotoba Composer
[![CI](https://github.com/OpenAI/modular_composer/actions/workflows/ci.yml/badge.svg)](https://github.com/OpenAI/modular_composer/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/OpenAI/modular_composer/branch/main/graph/badge.svg)](https://codecov.io/gh/OpenAI/modular_composer)
[![python-tests](https://github.com/OpenAI/modular_composer/actions/workflows/python-tests.yml/badge.svg)](https://github.com/OpenAI/modular_composer/actions/workflows/python-tests.yml)
[![Nightly](https://github.com/OpenAI/modular_composer/actions/workflows/nightly-bench.yml/badge.svg)](https://github.com/OpenAI/modular_composer/actions/workflows/nightly-bench.yml)
[![PyPI](https://img.shields.io/pypi/v/modular-composer.svg)](https://pypi.org/project/modular-composer/)
[![Contributing](https://img.shields.io/badge/CONTRIBUTING-guide-blue.svg)](CONTRIBUTING.md)
[![Plugin Build](https://github.com/OpenAI/modular_composer/actions/workflows/plugin.yml/badge.svg)](https://github.com/OpenAI/modular_composer/actions/workflows/plugin.yml)
[![GUI Build](https://github.com/OpenAI/modular_composer/actions/workflows/gui.yml/badge.svg)](https://github.com/OpenAI/modular_composer/actions/workflows/gui.yml)


This project blends poetic Japanese narration with emotive musical arrangements.

It automatically generates chords, melodies and instrumental parts for each chapter of a text, allowing verse, chorus and bridge sections to be arranged with human‑like expressiveness.

## Table of Contents
- [Setup](#setup)
- [Configuration Files](#configuration-files)
- [Generating MIDI](#generating-midi)
- [Batch audio-to-MIDI conversion](#batch-audio-to-midi-conversion)
- [Duration CSV Extraction](#duration-csv-extraction)
- [Breath Control](#breath-control)
- [Demo MIDI Generation](#demo-midi-generation)
- [Notebook Demo](#notebook-demo)
- [Tone and Dynamics](docs/tone.md)
- [Humanizer Reference](docs/humanizer.md)
- [Late-Humanize & Leak Jitter](docs/humanizer.md#late-humanize)
- [Groove Enhancements](docs/groove.md)
- [Phrase Diversity](docs/diversity.md)
- [Strings Articulations](docs/strings_generator.md)
- [Effects & Automation](docs/effects.md)
- [Vocal Generator](docs/vocal_generator.md)
- [Sax Generator](docs/sax_generator.md)
- [Realtime WebSocket Streaming](docs/realtime_ws.md)
- [Plugin & GUI](docs/plugin_gui.md)


## Setup
Before running any tests or generation scripts you must install the project dependencies.  Execute

```bash
bash setup.sh
```
For lightweight tests run:
```bash
LIGHT=1 bash setup.sh
```

or equivalently

```bash
pip install -r requirements/base.txt  # + optional extras
pip install -r requirements/extra-ml.txt
pip install -r requirements/extra-audio.txt
pip install -e .[gui]                 # optional GUI
```

See [v3 upgrade guide](docs/v3_upgrade.md) for migrating from the previous version.

### Quick Start

Install AI and audio extras for transformer-based generation:

```bash
pip install 'modular-composer[ai,audio]'
```

Install realtime dependencies:

```bash
pip install -e .[realtime]
```

This installs the same list as
[`requirements/realtime.txt`](requirements/realtime.txt).

`miditoolkit` will be used if `pretty_midi` is unavailable.


### Install from PyPI

```bash
pip install articulation-tagger==0.9.0
```

```python
from music21 import converter
from articulation_tagger import MLArticulationModel, predict_many

model = MLArticulationModel(num_labels=9)
scores = [converter.parse("a.mid"), converter.parse("b.mid")]
labels = predict_many(scores, model)
# **New split**
```bash
pip install -r requirements/base.txt -r requirements/extra-ml.txt -r requirements/extra-audio.txt

```
Run the REST API demo with Docker Compose:

```bash
docker compose up
```
ML 機能は PyTorch が必須です。

### フル機能を使うには

追加機能（RNN 学習や GUI、外部 MIDI 同期）を利用する場合は

```bash
pip install -r requirements/base.txt -r requirements/extra-ml.txt -r requirements/extra-audio.txt    # or: pip install 'modular-composer[rnn,gui,live]'
```

RNN features require `pip install 'modular-composer[rnn]'`.

Without these packages `pytest` and the composer modules will fail to import.

## Required Libraries
- **music21** – MIDI and score manipulation
- **pretty_midi** – MIDI export utilities (install via `[groove]`)
- **numpy** – numerical routines
- **PyYAML** – YAML configuration loader
- **pydantic** – configuration models
- **librosa** – WAV feature extraction (install via `[groove]`)
- **pydub** (optional) – audio post‑processing
- **mido** – MIDI utilities
- **scipy** – signal processing helpers
- **tqdm** – progress bars
- **colorama** – colored CLI output
- **tomli** – TOML parser
- **pytest** – test runner

For WAV file ingestion install the optional dependencies listed in
`requirements-optional.txt`.

The same list appears in [`requirements.txt`](requirements.txt) for reference.
Install the requirements before you invoke `modular_composer.py` or run
the tests—otherwise packages such as `music21` will not be available and
Python will raise a `ModuleNotFoundError`.

If you encounter `ModuleNotFoundError: No module named 'pkg_resources'` when
importing `pretty_midi`, install `setuptools` as well:

```bash
pip install setuptools
```

### Dev-Dependencies

The optional **Musyng Kite** SoundFont (LGPL) is recommended for audio previews.
Place the `.sf2` file somewhere and set the environment variable `SF2_PATH`
when rendering MIDI with `utilities.synth.render_midi`.

```bash
# preferred
bash setup.sh

# or equivalently
pip install -r requirements/base.txt
```

## Configuration Files
The `config/` directory stores YAML files that control generation.  The main entry is **`main_cfg.yml`**, which defines global tempo, key and paths to input data.  Example excerpt:

```yaml
# config/main_cfg.yml
global_settings:
  time_signature: "4/4"
  tempo_bpm: 88
  tempo_curve_path: "data/tempo_curve.json"  # optional gradual rit./accel.
  random_walk_step: 8  # ±8 range bar by bar
  # DrumGenerator.random_walk_step is deprecated; AccentMapper
  # now uses this value internally for both drums and bass.
  bass_range_hi: 64    # optional upper limit for bass notes (default 72)
paths:
  chordmap_path: "../data/processed_chordmap_with_emotion.yaml"
  rhythm_library_path: "../data/rhythm_library.yml"
  output_dir: "../midi_output"
sections_to_generate:
  - "Verse 1"
  - "Chorus 1"
```

Edit these values to point to your chordmap and rhythm library, and list the section labels you wish to render.
Chord progressions are defined in `utilities/progression_templates.yaml`. Append new progressions to this file and reload to use them without code changes.

[`data/tempo_curve.json`](data/tempo_curve.json) defines BPM over time. Each segment may specify
`"curve": "linear"` or `"step"` to control interpolation:

```json
[
  {"beat": 0, "bpm": 120, "curve": "linear"},
  {"beat": 32, "bpm": 108, "curve": "linear"},
  {"beat": 64, "bpm": 128}
]
```

## Generating MIDI
Before generating any MIDI ensure the requirements are installed with
`bash setup.sh` (or `pip install -r requirements/base.txt`).
Run the main script with the configuration file:

```bash
python modular_composer.py --main-cfg config/main_cfg.yml
```

By default the resulting MIDI is written to the directory specified by `paths.output_dir` in the config.  Use the `--dry-run` flag to skip the final export while still performing generation.
To change the drum mapping, pass `--drum-map` with one of the registered names such as `ujam_legend`:

```bash
python modular_composer.py --main-cfg config/main_cfg.yml --drum-map ujam_legend
```

This value can also be set via `global_settings.drum_map` in your configuration file.

Use `--strict-drum-map` if unknown drum instrument names should raise an error:

```bash
python modular_composer.py --main-cfg config/main_cfg.yml --strict-drum-map
```

You can override the guitar tuning directly from the CLI. Specify one of the
presets (`standard`, `drop_d`, `open_g`) or provide six comma-separated semitone
offsets:

```bash
python modular_composer.py --main-cfg config/main_cfg.yml --tuning drop_d
python modular_composer.py --main-cfg config/main_cfg.yml --tuning 0,-2,0,0,0,0
```

This overrides `part_defaults.guitar.tuning` in your configuration.

The same behaviour can be enabled with `global_settings.strict_drum_map: true` in your configuration file.

ベロシティフェードがフィル前の何拍に及ぶかを制御できます:

```yaml
global_settings:
  fill_fade_beats: 2.0 # デフォルトは2
```

### Batch audio-to-MIDI conversion

`utilities.audio_to_midi_batch` transcribes directories of stems into separate
single-track MIDI files. Use `--jobs` to process stems in parallel, but note
that each worker loads the CREPE model; on GPU this can quickly exhaust memory,
so large batches may require a smaller `--jobs` value. Non-WAV formats like
FLAC or MP3 rely on `librosa` and system codecs for decoding. `--resume`
maintains a log of completed stems, `--overwrite` forces re-transcription,
`--safe-dirnames` sanitizes song folder names, and `--merge` produces a single
multi-track MIDI per song.

```bash
python -m utilities.audio_to_midi_batch input/ output/ --ext wav,flac --resume
```

### Duration CSV Extraction

Use `utilities.duration_csv` to collect note durations, pitch and velocity from
MIDI files into a CSV. Pass `--instrument` to process only files whose
filenames contain the given string, matching case-insensitively.

```bash
python -m utilities.duration_csv data/midi --out data/duration/all.csv

# Only include files whose names contain "Guitar" (case-insensitive)
python -m utilities.duration_csv data/midi --out data/duration/guitar.csv --instrument guitar
```

## Breath Control

The rendering pipeline can process breaths automatically. Configure the
behaviour via `configs/render.yaml` or override from the CLI:

```bash
python scripts/render_audio.py voice.wav -o clean.wav --breath-mode remove \
  --log-level info
```
See [docs/render.md](docs/render.md) for the full option table.

Config keys:

| key | description | default |
| --- | ----------- | ------- |
| `breath_mode` | keep / attenuate / remove | `keep` |
| `attenuate_gain_db` | gain applied in attenuate mode | `-15` |
| `crossfade_ms` | remove mode crossfade length | `50` |
| `hop_ms` | analysis hop size | `10` |
| `thr_offset_db` | energy threshold offset | `-30` |
| `energy_percentile` | percentile for threshold | `95` |
| `log_level` | logging level | `WARN` |
Deprecated `breath_threshold_offset_db` in configs is still accepted but will emit a warning.

パターンオプションによるスタイルごとのオーバーライドは `options.fade_beats` を使います。

StudioOne labels C1 (MIDI 36) as B0. When exporting from that DAW the note
names may therefore appear one octave lower than the mapping used here.

DAWs sometimes label octaves differently. For instance StudioOne displays MIDI 36
as **B0** rather than **C1**, so exported UJAM patterns may look shifted even
though the notes are correct. You can switch mappings programmatically via
`utilities/drum_map_registry.get_drum_map`.


## Project Goal
"OtoKotoba" aims to synchronize literary expression and music.  Chapters of narration are mapped to emotional states so that chords, melodies and arrangements resonate with the text, ready for import into VOCALOID or Synthesizer V.

## Advanced Bass Features

| Feature               | Override Key              | Example                                          |
|-----------------------|---------------------------|--------------------------------------------------|
| Mirror vocal melody   | `mirror_melody`           | `mirror_melody: true`                            |
| Kick-lock velocity    | `velocity_shift_on_kick`  | `velocity_shift_on_kick: 12`                     |
| II–V build-up         | `approach_style_on_4th`   | `approach_style_on_4th: subdom_dom`              |
| Velocity envelope     | `velocity_envelope`       | `velocity_envelope: [[0.0,60],[2.0,90]]`         |

## Lyric-responsive drum fills

DrumGenerator now adjusts fill density based on the emotional intensity of each
section.  The mapping from intensity (0–1) to fill density can be customized via
`drum.fill_density_lut` in the YAML configuration.  Higher intensity sections
produce richer fills automatically.

### Adjust drum fill density

Edit `config/drum_settings.yaml` to fine‑tune fill density:

```yaml
drum:
  fill_density_lut:
    0.0: 0.05
    0.2: 0.10
    0.5: 0.30
    0.8: 0.48
    1.0: 0.65
```
Call `reload_lut()` on an existing `DrumGenerator` to apply edits without restarting.

## Advanced Guitar Features

| Feature | Parameter | Effect |
|---------|-----------|--------|
| Stroke direction | `stroke_direction` | `"down"` multiplies velocity by 1.1, `"up"` by 0.9 |
| Palm mute | `palm_mute` | Shortens sustain by 15% and lowers velocity |
| Slide timing | `slide_in_offset`, `slide_out_offset` | Fractional offsets (0.0–1.0) describing portamento start and end |
| Fret bend | `bend_amount`, `bend_release_offset` | Bend depth in semitones and release position before note end |
| Fingering controls | `position_lock`, `preferred_position`, `open_string_bonus`, `string_shift_weight`, `fret_shift_weight`, `strict_string_order` | Defaults: `False`, `0`, `-1`, `2`, `1`, `False` |
| Percussion n-gram | `parts.percussion.model_path` | Path to n‑gram model for auxiliary percussion |

Percussion hits that land on the same tick as a kick or snare are delayed by one tick when merged.

## Velocity presets by tuning

Provide a YAML or JSON file containing velocity curves for each tuning and style:

```yaml
standard:
  default: [40,50,65,80,96,112,124]
  power_chord: [45,55,70,85,100,118,127]
drop_d:
  default: [38,48,60,75,92,108,122]
```

Specify the file path via `velocity_preset_path` when instantiating `GuitarGenerator`.
The generator chooses the preset matching its tuning name and style; if absent,
a rounded fallback curve is generated.

### Amp presets

Set an amp model per section using `amp_preset`.  Preset values, effect levels
and cabinet IRs are loaded from `data/amp_presets.yml` by default:

```yaml
part_params:
  guitar:
    amp_preset: drive
```
The preset file now defines CC levels per amp model:

```yaml
presets:
  drive: 90
levels:
  drive: {reverb: 60, chorus: 45, delay: 30}
```
When you later call `export_audio()` the selected IR file will be used
automatically if `part.metadata.ir_file` is present.

To convolve the rendered WAV offline run the helper script:

```bash
python cli/ir_render.py dry.wav drive -g 3 -l -14 -b 16384
```

See [Effects and Automation](docs/effects.md#オフライン-ir-レンダリング) for details.

## Humanize – intensity envelope / swing override

Velocity scaling now follows each section’s `musical_intent.intensity`.
Bass patterns map a velocity tier (`low`, `mid`, `high`) to concrete MIDI ranges.
When `swing_ratio` is set, even eighth-notes are delayed by that amount and the
preceding note trimmed so the bar length remains intact.

## Phase 9 – Flexible Phrases

Bass generation now supports arbitrary time signatures via `--time-signature`.
Specify phrases and intensities in YAML and load them with `--phrase-spec`.
Custom templates can be inserted at sections using `--insert-phrase fill1@bridge`.
Example:

```bash
python modular_composer.py --main-cfg config/main_cfg.yml \
    --time-signature 7/8 --phrase-spec spec.yml \
    --insert-phrase my_fill@bridge
```


## Demo MIDI Generation

After fixing drum pattern durations you can generate test MIDIs with the helper
script:

```bash
bash run_generate_demo.sh
```

Alternatively run `make` directly:

```bash
bash -c "make demo && echo 'OK'"
```

## Velocity Random Walk debug CC

Enable `export_random_walk_cc: true` under `global_settings` to export the
random walk value as MIDI CC20 once per bar. The CC value is scaled so 64
represents no offset and values stay within the 0–127 range.
The bar's absolute start offset (`bar_start_abs_offset`) is passed through so timings match the score.

If the command finishes without errors you should see the message:

```
drum_patterns の duration 欠損が解消されました
```

## 標準パターン拡充

[`data/drum_patterns.yml`](data/drum_patterns.yml) に `tom_dsl_fill` タイプのフィルを追加しました。以下のように簡潔な DSL でタム回しを記述できます。

```yaml
tom_run_short:
  description: "1 小節前半にタム回し"
  pattern_type: "tom_dsl_fill"
  length_beats: 1.0
  drum_base_velocity: 88
  pattern: |
    (T1 T2 T3 S)
```
`export_random_walk_cc: true` を設定すると、ランダムウォーク値を CC20 として書き出せます。

## Running Tests


Before running the tests make sure the requirements are installed:

```bash
bash setup.sh
```

With the dependencies available you can verify the build with:
Install the core requirements and the additional test packages before
running any tests:

```bash
pip install -r requirements/base.txt
pip install -r requirements-test.txt
```

Then verify the build with:

```bash
pytest -q
```
Run coverage with:

```bash
coverage run -m pytest --cov=models --cov=realtime
```

You can also run the suite via `tox` to test against multiple
Python versions if available:

```bash
tox -q
```

If you encounter an error mentioning `starlette.testclient` or `httpx`,
install the dev extras:

```bash
pip install -e .[dev]
```

Phase 3 のテストを実行するには次のように追加の依存関係を入れてください:

```bash
pip install -e '.[audio,test]'
```

Running the tests confirms that chord generation and instrument mappings behave as expected.

Golden MIDI regression files are stored as base64 text under [`data/golden/`](data/golden/).
Update them with:

```bash
UPDATE_GOLDENS=1 pytest tests/test_midi_regression.py
```

To render audio set `SF2_PATH` to your SoundFont and install `fluidsynth`.
Use `utilities.synth.render_midi` to convert MIDI files to WAV for quick checks.
For a short audio regression locally you can run:

```bash
sudo apt-get install fluidsynth timgm6mb-soundfont
python - <<'EOF'
from utilities.synth import render_midi
import pathlib, base64
tmp = pathlib.Path('tmp-local')
tmp.mkdir(exist_ok=True)
b64 = pathlib.Path('data/golden/rock_drive_loop.b64').read_text()
midi = tmp / 'rock_drive_loop.mid'
midi.write_bytes(base64.b64decode(b64))
render_midi(str(midi), 'rock_drive_loop.wav', soundfont='/usr/share/sounds/sf2/TimGM6mb.sf2')
EOF
```

**Spectral Regression:**
To detect subtle timbral changes, CI runs `pytest tests/test_audio_spectrum.py` comparing FFT magnitudes with a 5% tolerance.
Run it locally after generating snapshots in the `tmp/` directory with the audio regression step:

```bash
export SF2_PATH=sf2/TimGM6mb.sf2
pytest tests/test_audio_spectrum.py
```
Baseline snapshots are expected under [`data/golden/wav/`](data/golden/wav/). If they are
absent, the spectrum test will be skipped.

### Groove Sampler Usage

Train a groove model and generate MIDI directly via the CLI:

```bash
modcompose groove train data/loops --ext midi --out model.pkl
modcompose groove sample model.pkl -l 4 --temperature 0.8 --seed 42 > groove.mid
```

An RNN baseline is available for comparison:

```bash
modcompose rnn train loops.json --epochs 1 --out rnn.pt
modcompose rnn sample rnn.pt -l 4 > rnn.mid
```
Stream a trained model live:
```bash
modcompose live rnn.pt --backend rnn --bpm 100
```
Real-time audio requires the `sounddevice` backend and currently works on
Linux and macOS only.

### Real-time Streaming
Use the `realtime` backend to send MIDI to an external port in real time:

```bash
modular-composer live score.mid --backend realtime --port "IAC Driver Bus 1"
```
Omit `--port` to list available ports. Low latency is easiest to achieve with
the `python-rtmidi` backend. Set `MCY_USE_CYTHON=0` during installation if the
Cython build environment is unavailable.

Adjust scheduling jitter with ``--latency-buffer`` (milliseconds). Measure
actual latency after playback using ``--measure-latency``:

```bash
modular-composer live score.mid --backend realtime --port "IAC Driver Bus 1" \
  --latency-buffer 5 --measure-latency
```

For seamless streaming you can pre-generate upcoming bars:

```bash
modcompose live model.pt --backend rnn --buffer-ahead 4 --parallel-bars 2
```
Use `--threads N` or `--process-pool` with the `compose` command to enable
multi-threaded or multi-process generation.
See [performance tips](docs/performance.md) for more tuning options.

#### Quick preview
Deterministic sampling lets you audition a groove without randomness:

```bash
modcompose groove sample model.pkl -l 4 --temperature 0 --top-k 1 > beat.mid
```
Add ``--play`` for an instant listen. On Linux it tries ``timidity`` or ``fluidsynth``; on macOS ``afplay`` is used and on Windows ``wmplayer`` or ``start``:
```bash
modcompose groove sample model.pkl -l 1 --play
```
List auxiliary tuples without generating MIDI:
```bash
modcompose groove sample model.pkl --list-aux  # alias: -L or --aux-list
# with filtering
modcompose groove sample model.pkl --list-aux --cond '{"section":"chorus"}'
# toggle per-bar caching for profiling
modcompose groove sample model.pkl -l 8 --no-bar-cache
```

If no MIDI player is detected a warning is emitted and the raw MIDI is written to ``stdout``.

Generator fallback: if a drum part has an empty pattern and a groove model is
provided, a bar is sampled automatically so silent placeholders turn into
grooved backing.

### Training your first groove model

Prepare a loop cache for faster experiments:

```bash
modcompose loops scan data/loops --ext midi,wav --out loops.json --auto-aux
modcompose loops info loops.json
```

The ``--auto-aux`` option infers ``intensity`` and ``heat_bin`` from each loop.
Intensity is ``low`` when mean velocity is ``<=60``, ``mid`` for ``61-100`` and
``high`` above that. ``heat_bin`` is derived from the step with the most hits
using a 4-bit index.

WAV support requires `librosa`. Install via `pip install librosa` if you want to
include audio loops.

Groove Sampler **v1.1** supports auxiliary conditioning on section type,
heatmap bin and intensity bucket. Provide a JSON map at train time and pass
`--cond` when sampling. The JSON should map each loop file to its metadata:

```json
{
  "verse.mid": {"section": "verse", "heat_bin": 3, "intensity": "mid"},
  "chorus.mid": {"section": "chorus", "heat_bin": 7, "intensity": "high"}
}
```

Then train and sample as follows (``aux.json`` may also be ``aux.yaml``):

```bash
modcompose groove train data/loops --aux aux.json
modcompose groove sample model.pkl --cond '{"section":"chorus","intensity":"high"}' > groove.mid
```

If you omit `--aux` the model behaves like version 1.0.
See [docs/aux_features.md](docs/aux_features.md) for the schema specification.
Inspect a saved model with:

```bash
modcompose groove info model.pkl --json --stats
```

### Style/Aux Tagging

Version 2 extends auxiliary metadata with style and feel tags stored in
``.meta.yaml`` files next to each loop. Train with ``--aux-key style`` and
sample with ``--cond-style`` or ``--cond-feel``:

```bash
python -m utilities.groove_sampler_v2 train data/loops/drums -o model.pkl --aux-key style
python -m utilities.groove_sampler_v2 sample model.pkl --cond-style lofi -l 4
```
For a quick textual overview you can also run:
```bash
modcompose groove info model.pkl --stats
```
This displays the model order, auxiliary tuples, token counts per instrument,
and the serialized size. ``groove info --stats`` also prints the token count
and training perplexity along with a short ``sha1`` hash derived from the pickle
payload so you can quickly compare models.

Order can be selected automatically using minimal perplexity on a validation
split. The CLI exposes smoothing parameters as well. Use ``--alpha`` to control
additive smoothing strength and ``--discount`` for Kneser–Ney:

```bash
modcompose groove train loops/ --ext wav,midi --order auto \
    --smoothing add_alpha --alpha 0.1 --out model.pkl
```

Kneser–Ney smoothing often yields lower perplexity on sparse or highly
heterogeneous data. A discount around ``0.75`` works well in most cases:

```bash
modcompose groove train loops/ --ext wav,midi --order auto \
    --smoothing kneser_ney --discount 0.75 --out model.pkl
```

### Humanise

Add subtle velocity and timing variation using the trained histograms:

```bash
modcompose groove sample model.pkl -l 8 \
    --humanize vel,micro --micro-max 24 --vel-max 45 > groove.mid
```
Velocity histograms can further refine dynamics:

```bash
modcompose render spec.yml --velocity-hist groove_hist.pkl \
    --humanize-velocity 1.0 --ema-alpha 0.2 --humanize-timing 1.0 --seed 42
```
Specifying ``--seed`` makes velocity sampling reproducible.

### Velocity Model Training

Use the ``train-velocity`` script to fit a simple KDE-based velocity model:

```bash
train-velocity --epochs 5 --out checkpoints/last.ckpt
```

### Velocity CLI Commands

| Command | Purpose |
| ------- | ------- |
| `train-velocity build-velocity-csv` | Scan MIDI tracks and drums to create a velocity CSV dataset. |
| `train-velocity augment-data` | Augment WAV loops and rebuild the CSV file. |
| `train-velocity` | Train the ML velocity model from a CSV file. |

**Key flags**

- `--csv-path` – path to the training CSV file.
- `--augment` – enable on-the-fly augmentation during training.
- `--seed` – RNG seed for reproducible runs.

See [docs/ml_velocity.md](docs/ml_velocity.md) for advanced settings.

### Sampling API

The helper ``generate_bar`` yields one bar at a time and updates the history
list in-place:

```python
from utilities import groove_sampler_ngram as gs
model = gs.load(Path("model.pkl"))
hist: list[gs.State] = []
events = gs.generate_bar(hist, model=model, temperature=0.0, top_k=1)
```

Deterministic generation can be achieved by setting ``temperature`` to ``0``
and ``top_k`` to ``1``:

```python
events = gs.generate_bar(hist, model=model, temperature=0, top_k=1)
```

You may constrain choices to the top ``k`` states and condition on auxiliary
labels such as section or intensity:

```python
events = gs.generate_bar(
    hist,
    model=model,
    temperature=0.8,
    top_k=3,
    cond={"section": "chorus", "intensity": "high"},
)
```

Passing ``temperature=0`` selects the most probable state deterministically.
Use ``--humanize vel,micro`` when sampling from the CLI to apply velocity and
micro‑timing variation.

### DAW Usage

Import the resulting ``groove.mid`` into your DAW (Ableton, Logic, etc.).
Velocity humanisation stays within MIDI 1–127 while micro timing
deviations are clipped to ± ``micro_max`` ticks (default 30) so alignment remains manageable.

### Groove Sampler v2

Build and sample using the optimized model:

```bash
python -m utilities.groove_sampler_v2 train data/loops -o model.pkl \
    --auto-res --jobs 8 --memmap-dir mmaps
python -m utilities.groove_sampler_v2 sample model.pkl -l 4 \
    --temperature 0.8 --cond-velocity hard --seed 42
```

If `librosa` is installed, training will auto‑detect the tempo of WAV loops;
otherwise it falls back to 120 BPM. Pass `--fixed-bpm` to override this
behaviour.

### Latency Benchmarks

| Model | Avg Latency per bar |
|-------|--------------------|
| n-gram | < 5 ms |
| RNN    | < 10 ms |

Launch the Streamlit GUI to compare:

```bash
modcompose gui
```
Refer to [docs/gui.md](docs/gui.md) for the new MIDI capture and preset features.

### RNN Backend and Live Playback

Train a simple recurrent model and stream it live:

```bash
modcompose rnn train loops/ -o rnn.pt
modcompose live rnn.pt --backend rnn --bpm 110
```
Pass `--sync external` to follow an external MIDI clock. This requires an
available MIDI-IN port provided by `mido`.

## Vocal Sync


Run this command to extract amplitude peaks from your narration. The peaks are
saved to JSON so they can be used for later synchronization tools:

```bash
modcompose peaks path/to/vocal.wav -o peaks.json --plot
```

Alternatively you can invoke the extraction helper directly:

```bash
python -m utilities.consonant_extract path/to/vocal.wav -o peaks.json
```

To use the Essentia backend:

```bash
modcompose peaks vocal.wav -o peaks.json --algo essentia
```

Use the JSON with the sampler to synchronise drums with consonants:

```bash
modcompose sample model.pkl --peaks peaks.json --lag 10
```

`global_settings.use_consonant_sync` enables this alignment. Set
`consonant_sync_mode` to control how strictly events follow detected consonants.
In **`bar`** mode the whole bar shifts toward the nearest consonant cluster,
whereas **`note`** mode aligns kick and snare hits individually. The default is
`bar` as shown in [config/main_cfg.yml](config/main_cfg.yml):

```yaml
global_settings:
  use_consonant_sync: true
  consonant_sync_mode: bar  # 'bar' or 'note'
consonant_sync:
  note_radius_ms: 30.0
  velocity_boost: 6  # set return_vel=True when using align_to_consonant directly
```

You can override this on the command line:

```bash
python modular_composer.py --main-cfg config/main_cfg.yml --consonant-sync-mode note
```

## Auto-Tag & Augmentation

Automatically infer section and intensity labels for your loop library:

```bash
modcompose tag loops/ --out meta.json --k-intensity 3 --csv summary.csv
```

This writes per-bar metadata to `meta.json` and a flat CSV summary. Use the augmentation
tool to apply swing, shuffle and transposition before training:

```bash
modcompose augment in.mid --swing 54 --transpose 2 -o out.mid
```

Combine both with the training commands via `--auto-tag`.

## GUI v2 Walkthrough

Launch the updated Streamlit interface:

```bash
modcompose gui
```

Upload a model in the sidebar, choose backend and bars to generate, then select the
desired section and intensity from the dropdowns populated by the model metadata. Click
"Generate" to view a pianoroll heatmap and audition the groove directly in the browser.

Passing `--lag` values below zero will pre-hit the drums. If this causes
negative beat offsets, set `clip_at_zero=true` in your configuration or pass the
parameter when using the synchroniser programmatically.

### Render from a Score Spec

Generate a simple MIDI file from a YAML or JSON description:

```bash
modcompose render spec.yml -o out.mid --soundfont path/to/timgm.sf2 \
  --normalize-lufs -14
```

### Golden MIDI Regression

Check serialized MIDI files for unwanted changes:

```bash
modcompose gm-test tests/golden_midi/*.mid
```

MIDI events are normalised before comparison so header metadata can vary.
Add `--update` to overwrite the expected files after intentional changes.

This JSON can then be fed to later synchronization tools.

## Bass Generator Usage

Bass lines can be generated directly from an emotion profile. The YAML file
[`data/emotion_profile.yaml`](data/emotion_profile.yaml) defines riffs per emotion. Render a bass part
locked to kick drums:

```python
from music21 import instrument
from generator.bass_generator import BassGenerator

gen = BassGenerator(
    part_name="bass",
    default_instrument=instrument.AcousticBass(),
    global_tempo=120,
    global_time_signature="4/4",
    global_key_signature_tonic="C",
    global_key_signature_mode="major",
    emotion_profile_path="data/emotion_profile.yaml",
)
section = {
    "emotion": "joy",
    "key_signature": "C",
    "tempo_bpm": 120,
    "chord": "C",
    "melody": [],
    "groove_kicks": [0, 1, 2, 3],
}
part = gen.render_part(section)
```

### Emotion Profile Format

[`data/emotion_profile.yaml`](data/emotion_profile.yaml) maps emotion names to
generator settings. Each entry must provide:

* `bass_patterns` – riffs with optional velocity and swing hints
* `octave_pref` – preferred octave region (`low`, `mid` or `high`)
* `length_beats` – number of beats the pattern spans

Generators look up the current section's emotion and apply these values when
creating parts.

```yaml
joy:
  bass_patterns:
    - riff: [1, b3, 5, 6]
      velocity: mid
      swing: off
  octave_pref: mid
  length_beats: 4
```

### Kick-Lock → Mirror-Melody

The first beat snaps to the nearest kick within the opening eighth note, then
the bass mirrors the lead melody around the chord root.

### ii–V Build-up

When the upcoming bar resolves back to the song's tonic, `render_part()` will
walk up the last two beats to lead into that cadence. Beats one and two still
use Kick‑Lock → Mirror‑Melody while beats three and four outline the ii or V
approach.

```python
next_sec = {"chord": "Cmaj7"}
part = gen.render_part({"chord": "G7", "groove_kicks": [0], "melody": []},
                       next_section_data=next_sec)
```

## Hi-Fi RNN Backend

Groove generation can now leverage a Lightning-based RNN with attention. Train a
model using:

Install extras via `pip install 'modular-composer[rnn]'` to enable this baseline.

```bash
modcompose rnn train loops.json --epochs 10 --out model.pt
```

Sample with:

```bash
modcompose rnn sample model.pt -l 4 > pattern.json
```

## AI Bass Generator

See [docs/ai_generator.md](docs/ai_generator.md) for advanced usage.
Install `transformers` and `torch` to experiment with a language-model driven bass line
generator. Pass `--backend transformer` and specify the model name and optional rhythm schema.

| Token | フィール | 説明 |
|-------|---------|------|
| `<straight8>` | ストレート 8分 | 基本的な 8 分刻み |
| `<swing16>`   | スウィング 16分 | 軽い跳ね感 |
| `<shuffle>`   | シャッフルフィール | 複雑な 3 連グルーヴ |

```bash
modcompose live model.pkl --backend transformer --model-name gpt2-medium --rhythm-schema <straight8>
```

Historical generation data can guide future runs when `--use-history` is set.
See [docs/ai.md](docs/ai.md) for details.

Interactive usage:

```bash
modcompose interact --backend transformer --model-name gpt2-medium \
  --midi-in "Device In" --midi-out "Device Out" --bpm 120 \
  --rhythm-schema <swing16>
```

## Tone and Dynamics

Use articulation key switches and amp presets to refine playback. Add
`--articulation-profile` to `compose` commands to load a YAML mapping.
Audio rendered with `modcompose render` can be loudness normalised via
`--normalize-lufs`.
### Realtime Options

Common CLI options:

- `--late-humanize` shifts note timing a few milliseconds right before playback.
- `--rhythm-schema` prepends a rhythm style token when sampling transformer bass.
- `--normalize-lufs` normalises rendered audio to the given loudness target.
- `normalize_wav` can also infer targets per section using
  `{'verse': -16, 'chorus': -12}`.
- `--buffer-ahead` and `--parallel-bars` control the pre-generation buffer for
  live mode. Increase them if generation is slow.
- ToneShaper selects amp presets using both intensity and average note
  velocity, then emits CC31 at the start of each part. Use it automatically at
  the end of `BassGenerator.compose()`:

  ```python
  from utilities.tone_shaper import ToneShaper

  shaper = ToneShaper()
  preset = shaper.choose_preset(intensity="medium", avg_velocity=avg_vel)
  part.extra_cc.extend(
      shaper.to_cc_events(amp_name=preset, intensity="medium", as_dict=True)
  )
  ```

Run with automatic tone shaping:

```bash
modcompose render spec.yml --tone-auto
```

`modcompose sample` accepts `--tone-preset` to select one of the built-in piano
profiles (`grand_clean`, `upright_mellow`, `ep_phase`). Vocal articulation is
enabled by default; pass `--no-enable-articulation` to disable glissando and
trill tags. Generated notes are normalised with `normalize_velocities()` so that
loudness stays consistent.
See [docs/piano_delta.md](docs/piano_delta.md) for details.
Vocal vibrato may be tweaked via `--vibrato-depth` and
`--vibrato-rate` (cycles per quarter note):

```bash
modcompose sample model.pt --backend vocal --vibrato-depth 0.7 --vibrato-rate 6
```
Add to `main_cfg.yml` to avoid long CLI flags:
```yaml
part_defaults:
  piano:
    tone_preset: grand_clean
    enable_articulation: true
  vocal:
    vibrato_depth: 0.5
    vibrato_rate: 5.0
    enable_articulation: true
```


To emit CC11 and aftertouch for dynamic playback enable the flags programmatically:

```python
from utilities import humanizer

humanizer.set_cc_flags(True, True)
```

See [docs/tone.md](docs/tone.md) for details.
Realtime playback supports `--kick-leak-jitter` and `--expr-curve` to tweak
velocity curves.
See [docs/live_tips.md](docs/live_tips.md) for realtime options.

## Realtime Low-Latency

Live playback uses a double-buffered engine. Synchronise with external MIDI
clock using:

```bash
modcompose live model.pt --backend rnn --sync external --bpm 120 --buffer 2
```
You can inspect real-time jitter by passing ``--measure-latency`` when using the
``realtime`` backend.

## Notebook Demo

See [`notebooks/quick_start.ipynb`](notebooks/quick_start.ipynb) for a minimal walkthrough that trains a model and plays a short preview.

## Evaluation Metrics

Use the ``eval`` CLI to analyse MIDI files and model latency:

```bash
modcompose eval metrics in.mid
modcompose eval latency model.pkl --backend ngram
```

Metrics include swing accuracy, note density and velocity variance.
``BLEC`` (Binned Log-likelihood Error per Class) is computed as
``mean( KL(p_true || p_pred) / log(N) )`` where ``N`` is the number of bins.

## Effects & Rendering

Use ``modcompose fx render`` to convolve a MIDI file with an impulse response:

```bash
modcompose fx render song.mid --preset CRUNCH --out out.wav
```

Impulse responses distributed with this project are licensed under CC-BY 4.0.

## ABX Test

Launch a simple browser-based ABX comparison:

```bash
modcompose eval abx loops_human/ loops_ai/ --trials 12
```

The page relies on Tone.js for MIDI playback and records your score interactively.

## Advanced Generators

### Sax Generator

The sax backend improvises short solo phrases. Enable it by adding a `Sax Solo`
section in your configuration and run:

```bash
python modular_composer.py --main-cfg config/main_cfg.yml --dry-run
```

Important keys under `part_params.melody` include `seed`, `rhythm_key`, `growl`
and `altissimo`.

## PianoGenerator α: テンプレート伴奏

``PianoTemplateGenerator`` provides a minimal piano backing track generator used for quick demos.
Invoke it via the CLI:

```bash
modcompose sample dummy.pkl --backend piano_template
```

The generator outputs a simple root and shell voicing pattern and boosts velocities
around provided kick offsets.

## PianoGenerator β

Beta adds guide/drop2 voicing, pedal CCs, intensity control and an optional counter line.
Use the ``--voicing`` flag to select a mode:

```bash
modcompose sample dummy.pkl --backend piano_template \
  --voicing drop2 --intensity high --counterline -o piano.mid
```

The JSON output now includes ``hand`` and ``pedal`` fields.

#### Intensity & Density

| intensity | RH/LH note density |
|-----------|--------------------|
| low       | 50 % (sparse)      |
| medium    | 100 % (default)    |
| high      | 110 % + anticipation|

Adjust density with ``--intensity``.

``piano.anticipatory_chord`` in ``main_cfg.yml`` enables a short chord hit right before each vocal rest ends.

## PianoGenerator ML

Phase γ introduces a transformer-based voicing model.
Follow the quickstart below or see
[docs/piano_gamma.md](docs/piano_gamma.md) for details.

```bash
# extract events
python scripts/extract_piano_voicings.py --midi-dir midi/ --out piano.jsonl

# train the LoRA model
python train_piano_lora.py --data piano.jsonl --out piano_model --safe --eval
# auto scale hyperparams based on dataset size
python train_piano_lora.py --data piano.jsonl --out piano_model --auto-hparam

# sample with the ML backend
modcompose sample dummy.pkl --backend piano_ml --model piano_model --temperature 0.9
```
# 最小ステップ実行例 (テスト用)
python train_piano_lora.py --data piano.jsonl --out /tmp/piano_test --steps 1 --safe

--eval を使う場合は，下記のオプション実行後：
    pip install -r requirements/extra-ml.txt -r requirements/extra-audio.txt

![training](docs/img/piano_gamma_demo.png)


### Tokenizer export

```bash
python - <<'PY'
from transformer.tokenizer_piano import PianoTokenizer
tok = PianoTokenizer()
tok.export_vocab("models/vocab.json")
PY
```

## DAW Plugin Prototype

An experimental JUCE plugin bridges the Python engine via ``pybind11``.
Build it with:

```bash
modcompose plugin build --format vst3 --out build/
```

The plugin forwards host tempo to Python and streams the generated bar via a ring buffer.
CI builds the plugin on Linux and macOS; Windows builds are optional.

## WebSocket Bridge

Run a lightweight server that keeps the piano model warm and replies with the
next bar of tokens:

```bash
python -m realtime.ws_bridge
```

Send a JSON payload to `ws://localhost:8765` and receive the token list back:

```python
import asyncio, json, websockets

async def main():
    async with websockets.connect("ws://localhost:8765") as ws:
        await ws.send(json.dumps({"chord": [60, 64, 67], "bars_context": 2}))
        print(json.loads(await ws.recv()))

asyncio.run(main())
```

## Vocal Generator
**Vocal Articulation Flags**
- `--vibrato-depth X` (デフォルト 0.5)
- `--vibrato-rate Y` – 周波数 (四分音符あたりの周期数、デフォルト 5.0)
- `--no-enable-articulation` でビブラート／グリス／トリルを無効化
Lyrics can be supplied to `VocalGenerator.compose` via the `lyrics_words` option. Each syllable is greedily mapped to phonemes by `text_to_phonemes`. Notes longer than half a beat receive vibrato events generated by `generate_vibrato`. Convert a MIDI and phoneme JSON to WAV with:
```bash
python scripts/synthesize_vocal.py --mid vocal.mid --phonemes phon.json --out audio/
```

## TTS ONNX Integration
You can run the synthesizer with an ONNX model instead of the default TTS backend.
Enable verbose logging with `--log-level`:

```bash
python scripts/synthesize_vocal.py --mid vocal.mid --phonemes phon.json \
    --out audio/ --onnx-model model.onnx --log-level DEBUG
```

The script exits with code `0` on success and `1` on error.

Specify a custom phoneme mapping when sampling vocals:

```bash
modcompose sample --backend vocal --phoneme-dict custom_dict.json
```


## License
This project is licensed under the [MIT License](LICENSE).
