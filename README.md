# OtoKotoba Composer

This project blends poetic Japanese narration with emotive musical arrangements.  
It automatically generates chords, melodies and instrumental parts for each chapter of a text, allowing verse, chorus and bridge sections to be arranged with human-like expressiveness.

---

## Setup

1. **Create & activate a venv**  
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate


# OtoKotoba Composer

This project blends poetic Japanese narration with emotive musical arrangements.  
It automatically generates chords, melodies and instrumental parts for each chapter of a text, allowing verse, chorus and bridge sections to be arranged with human-like expressiveness.

---

## Setup

1. **Create & activate a venv**  
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate

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

### Bass advanced – mirror_melody
Set `mirror_melody: true` to invert the vocal melody when creating the bass line.
