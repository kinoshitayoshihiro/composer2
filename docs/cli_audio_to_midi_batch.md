# audio_to_midi_batch CLI

The `audio_to_midi_batch` utility converts directories of stem audio files into MIDI.

New flags for continuous control generation:

- `--emit-cc11/--no-emit-cc11` – enable or disable expression (CC11) curves.
- `--emit-cc64/--no-emit-cc64` – enable or disable sustain pedal (CC64) curves.
- `--controls-domain {time,beats}` – render curves in seconds or beats.
- `--controls-sample-rate-hz FLOAT` – sampling rate for generated curves.
- `--controls-res-hz FLOAT` – deprecated alias of `--controls-sample-rate-hz`.
- `--controls-post-bend {skip,add,replace}` – handle existing pitch bends before applying new curves.
- `--cc-strategy {energy,rms,none}` – source for CC11 dynamics.
- `--controls-channel-map "bend:0,cc11:0,cc64:0"` – route targets to MIDI channels.
- `--write-rpn-range/--no-write-rpn-range` – emit an RPN bend-range message once per channel (default on).
- `--controls-total-max-events INT` – proportional cap across all control events.

Deprecated flag mappings:

| Old flag | Replacement |
| --- | --- |
| `--controls-res-hz` | `--controls-sample-rate-hz` |
| `--controls-resolution-hz` | `--controls-sample-rate-hz` |
| `--write-rpn` | `--write-rpn-range` |

`--controls-post-bend` modes:

| Mode | Behaviour |
| --- | --- |
| `skip` | keep existing bends, ignore new ones |
| `add` | keep existing bends and add new ones |
| `replace` | drop existing bends before applying new ones |

Example (time-domain):

```bash
python -m utilities.audio_to_midi_batch input_dir midi_out   --emit-cc11 --emit-cc64 --controls-domain time --controls-sample-rate-hz 100   --controls-channel-map "bend:0,cc11:0,cc64:0" --controls-max-events 200   --write-rpn-range
```

Example (beats-domain with tempo map):

```bash
python -m utilities.audio_to_midi_batch input_dir midi_out   --controls-domain beats --tempo-map "[[0,120],[4,90]]"   --controls-sample-rate-hz 80 --controls-max-events 100
```

Beats-domain curves require a tempo map via `--tempo-map` or `--tempo-map-from-midi`.

Example with routing JSON:

```json
{
  "0": {"cc11": "expr.json", "bend": "vibrato.json"}
}
```

```bash
python -m utilities.audio_to_midi_batch input_dir midi_out   --controls-routing routing.json   --controls-args "--sample-rate-hz bend=80,cc11=30"
```
