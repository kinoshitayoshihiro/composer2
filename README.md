# OtoKotoba Composer
![CI](https://github.com/OpenAI/modular_composer/actions/workflows/ci.yml/badge.svg)
![Nightly](https://github.com/OpenAI/modular_composer/actions/workflows/nightly-bench.yml/badge.svg)


This project blends poetic Japanese narration with emotive musical arrangements.

It automatically generates chords, melodies and instrumental parts for each chapter of a text, allowing verse, chorus and bridge sections to be arranged with human‑like expressiveness.

## Table of Contents
- [Setup](#setup)
- [Configuration Files](#configuration-files)
- [Generating MIDI](#generating-midi)
- [Demo MIDI Generation](#demo-midi-generation)
- [Notebook Demo](#notebook-demo)


## Setup
Before running any tests or generation scripts you must install the project dependencies.  Execute

```bash
bash setup.sh
```

or equivalently

```bash
pip install -r requirements.txt      # core + music21
pip install -e .[gui]                # optional GUI
```

### フル機能を使うには

追加機能（RNN 学習や GUI、外部 MIDI 同期）を利用する場合は

```bash
pip install -r requirements-extra.txt    # or: pip install 'modular_composer[rnn,gui,live]'
```

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
pip install -r requirements.txt
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
`bash setup.sh` (or `pip install -r requirements.txt`).
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

The same behaviour can be enabled with `global_settings.strict_drum_map: true` in your configuration file.

ベロシティフェードがフィル前の何拍に及ぶかを制御できます:

```yaml
global_settings:
  fill_fade_beats: 2.0 # デフォルトは2
```

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

## Humanize – intensity envelope / swing override

Velocity scaling now follows each section’s `musical_intent.intensity`.
Bass patterns map a velocity tier (`low`, `mid`, `high`) to concrete MIDI ranges.
When `swing_ratio` is set, even eighth-notes are delayed by that amount and the
preceding note trimmed so the bar length remains intact.

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

Be sure you have installed the requirements via `bash setup.sh` (or
`pip install -r requirements.txt`) before running any tests.  Once the
packages are installed you can verify the build with:

```bash
pytest -q
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

### Latency Benchmarks

| Model | Avg Latency per bar |
|-------|--------------------|
| n-gram | < 5 ms |
| RNN    | < 10 ms |

Launch the Streamlit GUI to compare:

```bash
modcompose gui
```

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
modcompose render spec.yml -o out.mid --soundfont path/to/timgm.sf2
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

### Kick-Lock → Mirror-Melody

The first beat snaps to the nearest kick within the opening eighth note, then
the bass mirrors the lead melody around the chord root.
TODO: add GIF demo

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

```bash
modcompose rnn train loops.json --epochs 10 --out model.pt
```

Sample with:

```bash
modcompose rnn sample model.pt -l 4 > pattern.json
```

## Realtime Low-Latency

Live playback uses a double-buffered engine. Synchronise with external MIDI
clock using:

```bash
modcompose live model.pt --backend rnn --sync external --bpm 120 --buffer 2
```

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

## ABX Test

Launch a simple browser-based ABX comparison:

```bash
modcompose eval abx loops_human/ loops_ai/ --trials 12
```

The page relies on Tone.js for MIDI playback and records your score interactively.

## DAW Plugin Prototype

An experimental JUCE plugin bridges the Python engine via ``pybind11``.
Build it with:

```bash
modcompose plugin build --format vst3 --out build/
```

The plugin forwards host tempo to Python and streams the generated bar via a ring buffer.
CI builds the plugin on Linux and macOS; Windows builds are optional.

## License
This project is licensed under the [MIT License](LICENSE).
