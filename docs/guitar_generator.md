# GuitarGenerator Usage

## Velocity Curve Interpolation
A velocity curve may be provided as seven control points. When
:func:`utilities.velocity_curve.interpolate_7pt` is called the curve is
expanded to 128 values. By default the function uses a natural cubic spline
when SciPy is available. Pass ``mode="linear"`` or set
``velocity_curve_interp_mode="linear"`` on :class:`GuitarGenerator` to use
linear interpolation. The function always returns a list of 128 values.

## Tablature Export
`export_musicxml_tab` supports `xml`, `ascii`, and `lily` formats. ASCII and
LilyPond exports embed string and fret numbers so the output can be used with
common guitar editors. LilyPond export automatically inserts `\new TabStaff`
when a real `lilypond` executable is not found.

## Fingering Costs
Fingering behaviour can be tuned via the `FingeringCost` dataclass:

| Field | Default | Meaning |
|-------|---------|---------|
| `open_bonus` | `-1` | Bonus applied to open strings |
| `string_shift` | `2` | Cost for changing strings |
| `fret_shift` | `1` | Cost for moving fret numbers |
| `position_shift` | `3` | Additional cost when moving more than two frets |

Pass an instance or dictionary to `GuitarGenerator(fingering_costs=...)`.

## Stroke Velocity Factors
The default multipliers applied for pick strokes are:

| Stroke | Factor |
|--------|-------|
| DOWN   | 1.20  |
| UP     | 0.80  |

Custom values may be supplied via ``stroke_velocity_factor`` when creating
:class:`GuitarGenerator`.
