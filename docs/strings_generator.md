# StringsGenerator Articulations

Phase 1 introduces basic articulation support for the string ensemble generator.
Each rhythm event may specify an `articulations` list. Supported names are:
`sustain`, `staccato`, `accent`, `tenuto`, `legato`, `tremolo`, `pizz`, and
`arco`.

The value may be a list or a single string. When using a string,
multiple names can be joined with `+` or spaces:

```yaml
events:
  - duration: 1.0
    articulations: "staccato+accent"
```

The special name `sustain` clears any default articulations without adding
new markings.

Example section data:

```yaml
part_params:
  strings:
    default_articulations: ["pizz"]
```

```python
section["events"] = [
    {"duration": 1.0, "articulations": ["legato"]},
    {"duration": 1.0, "articulations": ["legato"]},
]
```

When two consecutive events specify `legato`, a single slur is created.  Default
articulations apply when an event omits the key.

## Phase 2 Options

StringsGenerator now supports velocity curves, timing jitter and bow position metadata.

- `default_velocity_curve`: a list of 3, 7 or 128 values describing a velocity
  mapping. Three-point curves are interpolated to 128 steps.
- `timing_jitter_ms`: maximum random offset in milliseconds.
- `timing_jitter_mode`: either `"uniform"` or `"gauss"`.
- `timing_jitter_scale_mode`: `"absolute"` (default) or `"bpm_relative"` which
  scales `timing_jitter_ms` relative to a reference BPM of 120.
- `balance_scale`: blend ratio for section dynamics. Lower values reduce
  contrast.
- `bow_position`: one of `tasto`, `normale` or `ponticello`.

## Phase 3 Options

Additional articulation and expression controls:

- Automatic slurs connect neighbouring notes when the interval is a second or
  smaller and both durations are at least `0.5` quarter lengths. Explicit
  `legato` articulations still take precedence and rests break the chain.
- `crescendo`: boolean flag to enable a default expression ramp over the section
  length.
- `dim_start` / `dim_end`: numeric CC11 values (1-127) defining a custom
  expression envelope. Values interpolate linearly from start to end across the
  section.

## Bow Position & Divisi

The optional `bow_position` field may be set per event or section. Values like
`"sul pont."` and `"sul tasto"` are recognized alongside the canonical names.
When the `divisi` option is enabled, supported modes are `"octave"` and
`"third"`; unknown strings default to a pitch a third above and emit a warning.

## Trill, Tremolo & Vibrato

Events may set `pattern_type` to `"trill"` or `"tremolo"`. A trill alternates
the base note with a transposed pitch while a tremolo rapidly repeats the same
pitch. The note spacing is derived from the specified `rate_hz` using the
formula ``60 / (tempo * rate_hz)``. Vibrato depth and speed can be provided via
``vibrato`` dictionaries either per-event or in `part_params`; generated notes
store the waveform in ``editorial.vibrato_curve``.

## Phase 4 Options

Expression maps bundle articulations, velocity curves and controller presets.
Maps are loaded from `data/expression_maps.yml` by default and selected per
section using `part_params.strings.expression_map`.

```yaml
gentle_legato:
  articulations: ["legato"]
  cc:
    1: 20
    11: [40, 80]
  velocity_curve_name: soft
```

When unspecified the generator falls back to a map based on emotion and
intensity, defaulting to `gentle_legato`.

Mapped articulations are appended to `default_articulations`. ``velocity_curve_name``
switches the internal curve via `resolve_velocity_curve`. CC values may be
single numbers or `[start, end]` lists which create linear envelopes across the
section.

The bow position is also emitted as CC71 (`tasto` → 20, `normale` → 64,
`ponticello` → 100). A ``mute`` flag in `style_params` sends CC20 at 127 when
enabled and 0 when disabled, reducing velocities slightly when no sampler
support is available.

``emotion_map`` entries map emotion/intensity pairs to expression map names. The
defaults are stored in the same YAML and can be overridden:

```yaml
emotion_map:
  default:
    default: gentle_legato
    high: marcato_drive
```

Controller numbers are configurable via `data/cc_map.yml`:

```yaml
expression: 11
mod: 1
bow_pressure: 2
bow_position: 71
mute_toggle: 20
```
