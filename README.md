# OtoKotoba Composer


This project blends poetic Japanese narration with emotive musical arrangements.

It automatically generates chords, melodies and instrumental parts for each chapter of a text, allowing verse, chorus and bridge sections to be arranged with human‑like expressiveness.

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
paths:
  chordmap_path: "../data/processed_chordmap_with_emotion.yaml"
  rhythm_library_path: "../data/rhythm_library.yml"
  output_dir: "../midi_output"
sections_to_generate:
  - "Verse 1"
  - "Chorus 1"
```

Edit these values to point to your chordmap and rhythm library, and list the section labels you wish to render.

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

After installing the requirements with `bash setup.sh` (or
`pip install -r requirements.txt`) you can verify the build by running:

```bash
pytest -q
```

Running the tests confirms that chord generation and instrument mappings behave as expected.

## License
This project is licensed under the [MIT License](LICENSE).
