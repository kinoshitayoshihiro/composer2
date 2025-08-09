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
| `cc64_ratio`* | float | Fraction of note duration with sustain pedal active |
| `cc11_onset`* | int | CC11 value at note onset |
| `cc11_mean`* | float | Mean CC11 value over the note duration |
| `bend`*      | int   | Pitch-bend value at onset (−8192…8191) |
| `bend_range`* | int   | Pitch-bend range in semitones (default ±2) |
| `bend_max_semi`* | float | Maximum absolute bend depth in semitones within the note |
| `bend_rms_semi`* | float | RMS bend depth in semitones |
| `vib_rate_hz`* | float | Estimated vibrato rate in Hz |

Columns marked with * are optional and can be omitted with `--no-cc` or
`--no-bend` when high-resolution controller or pitch-bend data is not needed.
CC11 columns are included only when `--include-cc11` is specified.

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
python -m utilities.rich_note_csv path/to/midi_dir --out notes.csv
python -m utilities.rich_note_csv --coverage notes.csv
```

This prints the percentage of non-null values for each column and helps verify
that the dataset is complete.

Quick pitch-bend inspection:

```
python -m utilities.rich_note_csv path/to/midi.mid | grep bend
```

