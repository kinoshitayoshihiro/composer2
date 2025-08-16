# Continuous Control Curves

The `ControlCurve` class and `apply_controls` helper render sparse automation into
MIDI control change (CC) and pitch-bend events.

## Pitch-bend range and RPN Null

By default `apply_controls` configures the pitch-bend range via the standard RPN
sequence `(101=0, 100=0, 6=MSB, 38=LSB)` and emits it once per channel. Some
external devices require an additional *RPN Null* to be sent after the range is
set. Pass `rpn_null=True` to `apply_controls` to append `(101=127, 100=127)` and
close the RPN session.

> **LSB encoding**
> The `RPN 0,0` LSB is commonly interpreted as 1/128th of a semitone. The
> optional `cents` mode provided by `write_bend_range_rpn` is for convenience
> and may not map perfectly on all devices.

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

| Flag | Description |
| ---- | ----------- |
| `--out OUT.mid` | write to a new file (default `out.mid`) |
| `--bend-range-semitones N` | configure pitch‑bend range |
| `--write-rpn` | emit RPN 0,0 range before bends |
| `--rpn-reset` / `--no-rpn-reset` | append or suppress the RPN Null |
| `--coarse-only` | omit the LSB for whole‑semitone ranges |
| `--lsb-mode {128th,cents}` | fractional range encoding; `cents` may not match all devices |
| `--ensure-zero-at-edges` / `--no-ensure-zero-at-edges` | force pitch‑bend back to 0 at curve boundaries |
| `--sample-rate-hz` | overrides like `bend=80,cc=30` |
| `--max-bend`, `--max-cc11`, `--max-cc64` | cap emitted events per target |
| `--value-eps`, `--time-eps` | de‑duplication thresholds |
| `--tempo-map` | tempo map as `FILE.py:FUNC` or JSON `[(beat,bpm),...]` |
| `--dry-run` | skip writing the output file but print a summary |

The tempo map may also be passed directly to `apply_controls` as a callable, a
mapping of channels to callables, or a nested mapping of channels and targets.

## Sampling rate

`ControlCurve` accepts `resolution_hz` controlling how densely the curve is
sampled. The older `sample_rate_hz` alias is still accepted but triggers a
`DeprecationWarning` and will be removed in a future release.

Recommended sampling rates are around ``20–50`` Hz for CC curves and ``50–100`` Hz for
pitch bends.

# Control Curves

`ControlCurve` provides a lightweight way to render controller and pitch‑bend data.
Both :py:meth:`ControlCurve.to_midi_cc` and :py:meth:`ControlCurve.to_pitch_bend`
**mutate** the passed :class:`pretty_midi.Instrument` in place and return ``None``.

## Targets

- **cc11** – expression (0–127)
- **cc64** – sustain pedal (0 off, 127 on)
- **bend** – pitch bend in semitones

## Pitch‑bend range RPN

When `write_rpn` is enabled the sequence `RPN 0,0` followed by `Data Entry` (MSB/LSB)
sets the bend range.  A final `RPN Null` (`101=127`, `100=127`) clears the selection
so subsequent RPN operations are unaffected.
`write_bend_range_rpn` offers `coarse_only=True` to emit only the MSB for
coarse semitone resolution, `lsb_mode` to choose between `"128th"` semitones
or `"cents"` for the LSB encoding, and `send_rpn_null` to control whether the
RPN Null is written.
**The `cents` LSB mode is not part of the MIDI specification; some devices may
interpret it differently.**

Pitch‑bend curves accept values in semitones (default) or normalized units
(`units="normalized"`) where `-1..1` maps to the full 14‑bit range.

## Domains and tempo

Curves may be defined in absolute seconds (`domain="time"`) or in beats
(`domain="beats"`).  For beat‑domain curves a tempo map can be supplied either as a
callable `beat→bpm` or as an event list `[(beat, bpm), ...]`.  The callable should
return the BPM at the queried beat.

```python
pm = pretty_midi.PrettyMIDI()
curve = ControlCurve([0, 1, 2], [0, 64, 127], domain="beats")
apply_controls(pm, {0: {"cc11": curve}}, tempo_map=[(0, 120), (1, 60)])
```

Here the first beat is rendered at 120 BPM and the second at 60 BPM. If a
`sample_rate_hz` is provided when rendering, resampling occurs before any
event‑thinning so ordering and endpoints remain intact.

`sample_rate_hz` resamples the curve via spline interpolation before any
event-thinning is applied.  The older `resolution_hz` alias remains for
backward compatibility but is deprecated and will be removed six months after
this release.

`apply_controls` exposes per-target `max_events`, `value_eps`, and `time_eps` to
limit or thin events after resampling. Endpoint samples are always preserved.
Negative values for `offset_sec` are clamped to `0.0` with a warning. A global
`controls_total_max_events` cap may be supplied to proportionally thin all
targets after per-target caps. `rpn_reset` appends the optional RPN Null
sequence, while `rpn_coarse_only` omits the LSB for whole‑semitone ranges.
Recommended starting values are `sample_rate_hz=20–50` for CC and `50–100` for
pitch bend with `max_events≈8–16` per curve.

## Examples

### CC11 crescendo
```python
pm = pretty_midi.PrettyMIDI()
inst = pretty_midi.Instrument(program=0)
curve = ControlCurve([0, 2], [0, 127])
curve.to_midi_cc(inst, 11)
```

### Vibrato bends
```python
pm = pretty_midi.PrettyMIDI()
inst = pretty_midi.Instrument(program=0)
curve = ControlCurve([0, 1], [0.0, 0.5])
apply_controls.write_bend_range_rpn(inst, 2.0)
curve.to_pitch_bend(inst, bend_range_semitones=2.0)
```

### Combined apply_controls example
```python
pm = pretty_midi.PrettyMIDI()
curve_cc = ControlCurve([0, 1], [0, 127])
curve_bend = ControlCurve([0, 1], [0.0, 1.0])
apply_controls(
    pm,
    {0: {"cc11": curve_cc, "bend": curve_bend}},
    max_events={"cc11": 8, "bend": 8},
)
```
