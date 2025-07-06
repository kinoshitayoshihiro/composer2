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
