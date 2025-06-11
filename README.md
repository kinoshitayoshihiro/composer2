# OtoKotoba Composer

This project blends poetic Japanese narration with emotive musical arrangements.  
It automatically generates chords, melodies and instrumental parts for each chapter of a text, allowing verse, chorus and bridge sections to be arranged with human‑like expressiveness.

## Required Libraries
- **music21** – MIDI and score manipulation
- **PyYAML** – YAML configuration loader
- **pretty_midi** – MIDI export utilities
- **pydub** (optional) – audio post‑processing

Install dependencies via pip:

```bash
pip install -r requirements.txt  # or install only the needed packages
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
Run the main script with the configuration file:

```bash
python modular_composer.py --main-cfg config/main_cfg.yml
```

By default the resulting MIDI is written to the directory specified by `paths.output_dir` in the config.  Use the `--dry-run` flag to skip the final export while still performing generation.

## Project Goal
"OtoKotoba" aims to synchronize literary expression and music.  Chapters of narration are mapped to emotional states so that chords, melodies and arrangements resonate with the text, ready for import into VOCALOID or Synthesizer V.
