# StringsGenerator Articulations

Phase 1 introduces basic articulation support for the string ensemble generator.
Each rhythm event may specify an `articulations` list. Supported names are:
`sustain`, `staccato`, `accent`, `tenuto`, `legato`, `tremolo`, `pizz`, and
`arco`.

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
