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
| `cc11_at_onset`* | int | CC11 value at note onset |
| `cc11_mean`* | float | Mean CC11 value over the note duration |
| `bend`*      | int   | Pitch-bend value at onset (−8191…8191) |
| `bend_range`* | int   | Pitch-bend range in semitones (default ±2) |
| `bend_max_semi`* | float | Maximum absolute bend depth in semitones within the note |
| `bend_rms_semi`* | float | RMS bend depth in semitones |
| `vib_rate_hz`* | float | Estimated vibrato rate in Hz |

Internal mapping: normalized ``[-1..1]`` ⇄ PB ``[-8191..+8191]``. MIDI raw:
``0..16383`` (center ``8192``); the signed convention is often shown as
``-8192..+8191``. We scale by ``PB_MAX`` (=8191) so that ±range maps exactly
to ±8191.

Columns marked with * are optional and can be omitted with `--no-cc` or
`--no-bend` when high-resolution controller or pitch-bend data is not needed.
CC11 columns are included only when `--include-cc11` is specified.
Future revisions will add explicit CC columns for automatically generated
expression and sustain data.

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

## Retrofitting continuous controls

Existing MIDI files can be enhanced with expression (CC11) and pitch bends via
`apply_controls_cli`:

```
python -m utilities.apply_controls_cli in.mid out.mid --controls bend:on,cc11:on --write-rpn
```

The batching tool `audio_to_midi_batch` can render these curves automatically
when invoked with `--controls`:

```
python -m utilities.audio_to_midi_batch input_dir output_dir --controls bend:on,cc11:on
```

Both interfaces accept flags such as `--controls-resolution-hz`,
`--controls-max-events`, and `--dedup-eps` to control sampling density, event
caps, and epsilon-based de-duplication.  Beat-domain rendering requires a tempo
map, either derived from the MIDI (`--tempo-map-from-midi`) or supplied via
`--tempo-map`.

