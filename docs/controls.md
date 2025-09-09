# Continuous Control Curves

The `ControlCurve` class and `apply_controls` helper render sparse automation into
MIDI control change (CC) and pitch-bend events. Events are routed per MIDI channel
by attaching them to instruments named `channel0`, `channel1`, … within the
`pretty_midi.PrettyMIDI` object; individual `ControlChange` and `PitchBend`
messages carry no explicit channel number.

## Pitch-bend range

`apply_controls` writes a bend‑range RPN `(101=0,100=0,6=MSB,38=LSB)` once per
instrument before any pitch bends when enabled. The library default emits this
sequence (`write_rpn=True`); the command‑line wrapper requires `--write-rpn`.
The LSB encodes the fractional portion in 1/128th‑semitone steps.

## Examples

Time‑domain rendering with RPN:

```bash
python -m utilities.apply_controls_cli song.mid --curves curves.json \
  --controls bend:on,cc11:on --write-rpn
```

Beats‑domain rendering (tempo map extracted from the MIDI):

```bash
python -m utilities.apply_controls_cli song.mid --curves curves.json \
  --controls bend:on --controls-domain beats \
  --tempo-map-from-midi song.mid --write-rpn
```

### Useful flags

| Flag | Default | Meaning |
| ---- | ------- | ------- |
| `--controls bend:on,cc11:off,cc64:off` | – | enable targets |
| `--controls-resolution-hz` | `100.0` | sampling rate of curves |
| `--controls-max-events` | `200` | cap events per curve |
| `--controls-total-max-events` | – | cap total control events |
| `--dedup-eps-time` | `1e-4` | merge events closer than this time Δ |
| `--dedup-eps-value` | `1.0` | merge events with small value Δ |
| `--bend-range-semitones` | `2.0` | pitch‑bend range | 
| `--sample-rate-hz` | overrides like `bend=80,cc=30` |
| `--max-bend`, `--max-cc11`, `--max-cc64` | cap emitted events per target |
| `--value-eps`, `--time-eps` | de‑duplication thresholds |
| `--dedup-eps-value`, `--dedup-eps-time` | pre-deduplication thresholds for knots |
| `--rpn-at` | timestamp for the bend-range RPN (default 0.0 seconds) |
| `--tempo-map` | tempo map as `FILE.py:FUNC` or JSON `[(beat,bpm),...]` |
| `--dry-run` | skip writing the output file but print a summary |

`--write-rpn` is a deprecated alias of `--write-rpn-range`.

The tempo map may also be passed directly to `apply_controls` as a callable, a
mapping of channels to callables, or a nested mapping of channels and targets.

## Sampling rate

`ControlCurve` accepts `sample_rate_hz` controlling how densely the curve is
sampled. The older `resolution_hz` alias is still accepted but triggers a
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
`write_bend_range_rpn` encodes the coarse semitone count via CC#6 and the
fractional remainder via CC#38. Setting `rpn_null=True` appends the RPN Null
sequence after data entry.

Pitch‑bend curves accept values in semitones (default) or normalized units
(`units="normalized"`) where `-1..1` maps to the full 14‑bit range.
The renderer automatically appends a final zero‑bend event so synths return to
pitch centre.

Values are quantized to the MIDI 14‑bit pitch‑bend range and clipped to
`-8191..+8191` (the minimum representable value is `-8191` due to rounding).

Internal mapping: normalized ``[-1..1]`` ⇄ PB ``[-8191..+8191]``. MIDI raw:
``0..16383`` (center ``8192``); the signed convention is often shown as
``-8192..+8191``. We scale by ``PB_MAX`` (=8191) so that ±range maps exactly to
±8191.

## Domains and tempo

Curves may be defined in absolute seconds (`domain="time"`) or in beats
(`domain="beats"`).  For beat‑domain curves a tempo map can be supplied either as a
callable `beat→bpm`, via `--tempo-map-from-midi`, or inline JSON with
`--tempo-map` such as `'[[0,120],[4,90]]'`.  The callable should return the BPM
at the queried beat.

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
Negative offsets are clamped to `0.0` with a warning. A global
Recommended starting values are `sample_rate_hz=20–50` for CC and
`50–100` for pitch bend with `max_events≈8–16` per curve.

Event timestamps are coerced to be strictly increasing (by at least
`time_eps`, default `1e-9`) to avoid zero-length deltas that some MIDI parsers
cannot handle.

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
