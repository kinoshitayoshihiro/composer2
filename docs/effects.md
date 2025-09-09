# Effects and Automation

ToneShaper presets are stored in `amp_presets.yml` using the format:

```yaml
presets:
  clean: 20
levels:
  clean: {reverb: 40, chorus: 20, delay: 10}
ir:
  clean: "irs/blackface-clean.wav"
```

Sections may include an `fx_envelope` describing mix automation
(older configs may use the alias `effect_envelope`):

```yaml
fx_envelope:
  0.0: {mix: 0.5}
  2.0: {mix: 1.0}
```

This envelope is converted to CC91/93/94 events via `ToneShaper.to_cc_events`.
By default CC91 controls reverb send, **93 controls delay**, and **94 controls chorus send**.
Each entry may specify a `cc` number directly or use a `type` shorthand (`reverb`, `delay`, `chorus`, `brightness`).
Valid `shape` values are `lin`, `exp`, and `log`.
MIDI values must be within `0-127`.

## FX Envelope Example

```yaml
fx_envelope:
  0.0:
    type: reverb
    start: 0
    end: 100
    duration_ql: 4.0
    shape: lin
  4.0:
    type: reverb
    start: 100
    end: 20
    duration_ql: 2.0
    shape: exp
```

### Brightness automation

```yaml
fx_envelope:
  0.0: {type: brightness, start: 40, end: 80, duration_ql: 2.0, shape: log}
```

## export_mix_json Example

`export_mix_json()` writes a JSON mapping of part IDs to mix data:

```json
{
  "g": {
    "extra_cc": [{"time": 0.0, "cc": 31, "val": 40}],
    "ir_file": "irs/blackface-clean.wav",
    "preset": "clean",
    "fx_cc": [{"time": 0.0, "cc": 91, "val": 60}]
  }
}
```

## オフライン IR レンダリング

`export_audio()` で生成した WAV ファイルは、IR (インパルス応答) を指定
することでオフラインで畳み込みを行えます。以下のように実行します。

```bash
python -m generator.guitar_generator --render section.yml --ir irs/blackface.wav
```
