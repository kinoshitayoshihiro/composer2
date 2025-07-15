# Quick Start

Install locally and run the training pipeline.

```bash
pip install -e .[dev]
python -m modcompose.auto_tag input.mid tags.yaml
python -m modcompose.train config.yaml
python -m modcompose.play live
```

Train the velocity model used for expressive playback:

```bash
train-velocity --help
```

For live playback, run:

```bash
modcompose realtime
```

### Section YAML fields

`load_chordmap` expects `sections` under `global_settings`. Missing
`vocal_midi_path` defaults to `<section>_vocal.mid` and all paths are resolved
relative to the YAML file.

```yaml
global_settings:
  tempo: 120
  sections:
    Verse:
      order: 1
      length_in_measures: 4
```

Generate a directory tree summary:

```bash
modcompose dump-tree <root> --version 3
```

### Arpeggio Pattern DSL

You can specify an arpeggio note order directly in your rhythm library using
`pattern_type: "arpeggio"`.

```yaml
guitar_arpeggio_basic:
  pattern_type: arpeggio
  string_order: [5,4,3,2,1,0,1,2,3,4]
  strict_string_order: true
  reference_duration_ql: 1.0
```

The guitar generator will strum notes sequentially according to
`string_order`.

### Fingering Options

`GuitarGenerator` exposes several parameters to control how fingering is
estimated:

- `position_lock`: restricts fingering around `preferred_position` (fret)
- `preferred_position`: center fret when `position_lock` is enabled
- `open_string_bonus`: negative cost encouraging open strings
- `string_shift_weight`: cost for changing strings between notes
- `fret_shift_weight`: cost for moving frets between notes
- `strict_string_order`: raise an error when the given `string_order` does not
  match the number of arpeggio notes

### Exporting Enhanced Tablature

After composing a guitar part you can write a simple tablature file:

```python
gen = GuitarGenerator(...)
part = gen.compose(section_data=some_section)
gen.export_tab_enhanced("out_tab.txt")
```

Each line of `out_tab.txt` contains the string number and fret for a note,
allowing quick import into external tools.

You can also export a MusicXML file containing the tablature markings:

```python
gen.export_musicxml_tab("out_tab.xml")
```

### Harmonics (Guitar)

```python
gen = GuitarGenerator(enable_harmonics=True, prob_harmonic=0.25)
```

You can disable harmonic generation from the CLI with `--no-harmonics`.

Set ``prob_harmonic`` to ``1.0`` to force harmonics for testing.  Use
``harmonic_types`` to filter between natural or artificial nodes.

### Harmonics (Strings)

```python
from generator.strings_generator import StringsGenerator
str_gen = StringsGenerator(enable_harmonics=True, max_harmonic_fret=15)
```

Both generators share the following loudness controls:

| Param | Usage |
|-------|-------|
| ``harmonic_volume_factor`` | Multiply velocity (0-1 range) |
| ``harmonic_gain_db`` | Override with dB adjustment |

``max_harmonic_fret`` sets the highest allowed touching fret.  MusicXML output
may require a notation program that understands harmonic notation.

### IR レンダリング

MIDI から直接 WAV を生成し、インパルス応答を適用するには `modcompose ir-render` を使用します。

```bash
modcompose ir-render part.mid irs/room.wav -o rendered.wav \
  --quality high --bit-depth 32 --oversample 2 --no-dither
```

Python からは ``GuitarGenerator.export_audio`` を使って IR 名を指定できます。

```python
gen.export_audio(ir_name="mesa412")
```

Tips: You can disable automatic down-mix with `downmix="none"` or keep the
default `"auto"`.

## Rendering quality matrix 📝

| Oversample factor | Bit-depth 16-bit | Bit-depth 24-bit | Bit-depth 32-float |
| :-- | :-- | :-- | :-- |
| 1× (Real-time) | ✔ 軽量 | — | — |
| 2× | ◎ 推奨 | ✔ | — |
| 4× | ◎ High-end | ◎ 推奨 | ✔ 研究用 |
| 8× | — | ◎ Audiophile | ✔ Mastering |

> **Tips:** Disable *dither* when `--no-normalize` is set, because dithering tries to mask quantization noise that would otherwise be inaudible. Without normalization, the added noise becomes unnecessary and may degrade the headroom.

### StringsGenerator Phase 2 Example

```yaml
part_params:
  strings:
    default_velocity_curve: [30, 80, 110]
    timing_jitter_ms: 20
    timing_jitter_scale_mode: bpm_relative
    bow_position: tasto
    macro_envelope:
      type: cresc
      beats: 4
      start: 40
      end: 90
```

### Strings IR レンダリング例

```python
from generator.strings_generator import StringsGenerator
gen = StringsGenerator()
parts = gen.compose(section_data={"section_name": "A", "q_length": 1.0})
gen.export_audio(ir_name="hall", out_path="out/strings_A.wav")
```

### Section YAML fields

Include per-section metadata in your chord map:

```yaml
sections:
  Verse:
    vocal_midi_path: "{section}_vocal.mid"
    consonant_json: verse_consonants.json
```

Paths resolve relative to `chordmap.yaml`.

### Dump-tree v3

Create a Markdown tree of any project directory:

```bash
modcompose dump-tree my_project --version 3
```

The file `tree.md` appears inside `my_project`.
