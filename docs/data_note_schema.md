# Data Note Schema

The project uses YAML files to describe song structure and emotion cues.  See
`generator/arranger.py` for a minimal example of how `chordmap.yaml`,
`rhythm_library.yaml` and `emotion_profile.yaml` interact to produce section-wise
MIDI parts.

# Rich Note CSV Schema

The `rich_note_csv.py` utility exports note-level information from MIDI files.
Each row corresponds to a single note with the following columns:

| Column       | Type  | Description |
|--------------|-------|-------------|
| `pitch`      | int   | MIDI note number 0–127 |
| `duration`   | float | Note length in seconds |
| `bar`        | int   | Zero-based bar index |
| `position`   | int   | Zero-based 16th-note slot inside the bar |
| `velocity`   | int   | MIDI velocity 0–127 |
| `chord_symbol` | str | Optional harmonic label |
| `articulation` | str | Optional articulation tag |
| `q_onset`    | float | Onset in quarter-note units |
| `q_duration` | float | Duration in quarter-note units |
| `CC64`*      | int   | Sustain pedal value (0–127) at onset |
| `bend`*      | int   | Pitch-bend value at onset (−8192…8191) |

Columns marked with * are optional and can be omitted with `--no-cc` or
`--no-bend` when high-resolution controller or pitch-bend data is not needed.

Bar and position derive from PrettyMIDI ticks:

```
ticks = pm.time_to_tick(note.start)
onset_quarter = ticks / pm.resolution
beats_per_bar = numerator * 4 / denominator
bar = floor(onset_quarter / beats_per_bar)
sixteenth = ticks * 4 // pm.resolution
position = sixteenth % (beats_per_bar * 4)
```

Currently only the initial time signature is considered if a MIDI file contains
multiple changes. Future versions may support per-bar time signature updates.

Coverage statistics for generated CSVs can be obtained with:

```
python -m utilities.rich_note_csv --coverage notes.csv
```

This prints the percentage of non-null values for each column and helps verify
that the dataset is complete.

