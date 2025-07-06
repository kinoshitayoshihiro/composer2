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
