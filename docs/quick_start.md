# Quick Start

Install locally and run the training pipeline.

```bash
pip install -e .[dev]
python -m modcompose.auto_tag input.mid tags.yaml
python -m modcompose.train config.yaml
python -m modcompose.play live
```

For live playback, run:

```bash
modcompose realtime
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
