# Continuous Control Curves

The `ControlCurve` class and `apply_controls` helper render sparse automation into
MIDI control change (CC) and pitch-bend events.

## Pitch-bend range and RPN Null

By default `apply_controls` configures the pitch-bend range via the standard RPN
sequence `(101=0, 100=0, 6=MSB, 38=LSB)` and emits it once per channel. Some
external devices require an additional *RPN Null* to be sent after the range is
set. Pass `rpn_null=True` to `apply_controls` to append `(101=127, 100=127)` and
close the RPN session.

## Routing example

Given a JSON routing description

```json
{"0": {"cc11": "expr.json", "bend": "vibrato.json"}}
```

you can render the controls into a MIDI file using the CLI (if installed):

```bash
python -m utilities.apply_controls song.mid routing.json
```

CLI flags:

- `--out OUT.mid` write to a new file (default: `out.mid`)
- `--bend-range-semitones N` configure pitch-bend range
- `--rpn-null` append an RPN Null after the range
- `--max-bend/--max-cc11/--max-cc64` cap event counts
- `--tempo-map FILE.py:FUNC` supply a beat→BPM callable for beats-domain curves
- `--dry-run` skip writing the output file

The tempo-map may also be passed directly to `apply_controls` as a callable, a
mapping of channels to callables, or a nested mapping of channels and targets.

## Sampling rate

`ControlCurve` accepts `sample_rate_hz` controlling how densely the curve is
sampled. The older `resolution_hz` alias is still accepted but triggers a
`DeprecationWarning` and will be removed in a future release.
