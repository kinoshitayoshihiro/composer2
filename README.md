# OtoKotoba Composer
![CI](https://github.com/OpenAI/modular_composer/actions/workflows/ci.yml/badge.svg)


This project blends poetic Japanese narration with emotive musical arrangements.

It automatically generates chords, melodies and instrumental parts for each chapter of a text, allowing verse, chorus and bridge sections to be arranged with human‑like expressiveness.

## Setup
Before running any tests or generation scripts you must install the project dependencies.  Execute

```bash
bash setup.sh
```

or equivalently

```bash
pip install -r requirements.txt
```

Without these packages `pytest` and the composer modules will fail to import.

## Required Libraries
- **music21** – MIDI and score manipulation
- **pretty_midi** – MIDI export utilities
- **numpy** – numerical routines
- **PyYAML** – YAML configuration loader
- **pydantic** – configuration models
- **pydub** (optional) – audio post‑processing
- **mido** – MIDI utilities
- **scipy** – signal processing helpers
- **tomli** – TOML parser
- **pytest** – test runner

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
paths:
  chordmap_path: "../data/processed_chordmap_with_emotion.yaml"
  rhythm_library_path: "../data/rhythm_library.yml"
  output_dir: "../midi_output"
sections_to_generate:
  - "Verse 1"
  - "Chorus 1"
```

Edit these values to point to your chordmap and rhythm library, and list the section labels you wish to render.

`data/tempo_curve.json` defines BPM over time. Each segment may specify
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
Overrides may specify `swing_ratio` to shift off-beats with a custom feel.

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

If the command finishes without errors you should see the message:

```
drum_patterns の duration 欠損が解消されました
```

## Running Tests

Be sure you have installed the requirements via `bash setup.sh` (or
`pip install -r requirements.txt`) before running any tests.  Once the
packages are installed you can verify the build with:

```bash
pytest -q
```

Running the tests confirms that chord generation and instrument mappings behave as expected.

Golden MIDI regression files are stored as base64 text under `data/golden/`.
Update them with:

```bash
UPDATE_GOLDENS=1 pytest tests/test_midi_regression.py
```

To render audio set `SF2_PATH` to your SoundFont and install `fluidsynth`.
Use `utilities.synth.render_midi` to convert MIDI files to WAV for quick checks.

### Groove Sampler Usage

Build an n-gram model from a folder of MIDI loops:

```bash
python -m utilities.groove_sampler data/loops --stats
```

The command prints detected resolution and the chosen order.

### Groove Sampler v2

Build and sample using the optimized model:

```bash
python -m utilities.groove_sampler_v2 train data/loops -o model.pkl \
    --auto-res --jobs 8 --memmap-dir mmaps
python -m utilities.groove_sampler_v2 sample model.pkl -l 4 \
    --temperature 0.8 --cond-velocity hard --seed 42
```

## Vocal Sync


Run this command to extract amplitude peaks from your narration. The peaks are
saved to JSON so they can be used for later synchronization tools:

```bash
modcompose peaks path/to/vocal.wav -o peaks.json --plot
```

Use the JSON with the sampler to synchronise drums with consonants:

```bash
modcompose sample model.pkl --peaks peaks.json --lag 10
```

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

## License
This project is licensed under the [MIT License](LICENSE).
