# OtoKotoba Composer

OtoKotoba Composer weaves narrated Japanese texts with emotive musical arrangements. It automatically generates chords, melodies and instrumental parts so each section of a story flows with human‑like expression.

## Setup

1. **Create and activate a virtual environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **(Optional) run the tests**
   ```bash
   pytest
   ```
4. **Generate MIDI output**
   ```bash
   python modular_composer.py --main-cfg config/main_cfg.yml
   ```

## Configuration Overview

`config/main_cfg.yml` controls global settings, file paths and which sections are rendered. A minimal example looks like this:

```yaml
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

## Drum Map & Velocity Curve

### Drum Map

The drum generator translates instrument names to MIDI note numbers using a *drum map*. By default it follows the General MIDI standard, but you can select another map or provide your own file.

```yaml
part_defaults:
  drums:
    drum_map: "GM"            # other examples: "Roland", "CustomMap"
```

### Velocity Curve

Rhythm patterns may include a `velocity_curve` option that gradually scales note velocities across the pattern.

```yaml
bass_patterns:
  bass_algo_desc_fifths_quarters:
    options:
      note_resolution_ql: 1.0
      velocity_curve: "crescendo"  # fades from soft to strong
```
