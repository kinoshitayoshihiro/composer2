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
- `divisi` – `bool` or mapping enabling octave or third splits for Violin I/II.
