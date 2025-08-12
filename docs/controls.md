# Control Curves

`ControlCurve` provides a lightweight way to render controller and pitch‑bend data.

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

Pitch‑bend curves accept values in semitones (default) or normalized units
(`units="normalized"`) where `-1..1` maps to the full 14‑bit range.

## Domains and tempo

Curves may be defined in absolute seconds (`domain="time"`) or in beats
(`domain="beats"`).  For beat‑domain curves a tempo map can be supplied either as a
callable `beat→bpm` or as an event list `[(beat, bpm), ...]`.

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
backward compatibility but is deprecated and will be removed six months
after this release.

`apply_controls` exposes `cc_max_events` / `bend_max_events` to limit emitted
events and `value_eps` / `time_eps` to adjust de‑duplication sensitivity.
Negative values for `offset_sec` are clamped to `0.0` with a warning. Recommended
starting values are `sample_rate_hz=20–50`, `cc_max_events=8–16`, and
`bend_max_events=8–16`.

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
    cc_max_events=8,
    bend_max_events=8,
)
```
