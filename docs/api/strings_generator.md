# StringsGenerator API Reference

The `StringsGenerator` class creates simple block-chord parts for a standard
string ensemble. It inherits from `BasePartGenerator` and exposes two main
methods. Additional options allow basic smart voicing and divisi handling.

## Methods

### `compose(section_data: dict) -> dict[str, music21.stream.Part]`
Generates five ``Part`` objects (Contrabass, Violoncello, Viola, Violin II,
Violin I) from a single chord symbol.

### `export_musicxml(path: str)`
Exports the previously generated parts as one `Score` in the order listed
above.

### Parameters
- `voicing_mode` – ``"close"`` (default), ``"open"`` or ``"spread```` for the
  voicing density.
- `voice_allocation` – optional mapping of chord tone index per section; use
  ``-1`` to silence a section.
- `divisi` – `bool` or mapping enabling octave or third splits for Violin I/II. Use `"third"` to add a diatonic third above; if the added third exceeds the instrument range it will drop an octave or be omitted.
- `avoid_low_open_strings` – when ``True`` transposes viola/violin notes on open C/G strings up an octave.
