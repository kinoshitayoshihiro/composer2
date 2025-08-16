# audio_to_midi_batch CLI

The `audio_to_midi_batch` utility converts directories of stem audio files into MIDI.

New flags for continuous control generation:

- `--emit-cc11/--no-emit-cc11` – enable or disable expression (CC11) curves.
- `--emit-cc64/--no-emit-cc64` – enable or disable sustain pedal (CC64) curves.
- `--controls-domain {time,beats}` – render curves in seconds or beats.
 - `--controls-sample-rate-hz FLOAT` – sampling rate for generated curves.
 - `--controls-res-hz FLOAT` – deprecated alias of `--controls-sample-rate-hz`.
 - `--controls-max-events INT` – cap the number of events per curve.
 - `--controls-total-max-events INT` – global cap across all generated CC and bend events.
- `--cc-strategy {energy,rms,none}` – source for CC11 dynamics.
- `--controls-channel-map "bend:0,cc11:0,cc64:0"` – route targets to MIDI channels.
- `--write-rpn-range/--no-write-rpn-range` – emit an RPN bend-range message once per channel.

Example (time-domain):

```bash
python -m utilities.audio_to_midi_batch input_dir midi_out \
  --emit-cc11 --emit-cc64 --controls-domain time --controls-sample-rate-hz 100 \
  --controls-channel-map "bend:0,cc11:0,cc64:0" --controls-max-events 200 \
  --controls-total-max-events 500 \
  --write-rpn-range
```

Example (beats-domain with tempo map):

```bash
python -m utilities.audio_to_midi_batch input_dir midi_out \
  --controls-domain beats --tempo-map "[[0,120],[4,90]]" \
  --controls-sample-rate-hz 80 --controls-max-events 100
```
