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

Sections may include an `fx_envelope` describing mix automation:

```yaml
fx_envelope:
  0.0: {mix: 0.5}
  2.0: {mix: 1.0}
```

This envelope is converted to CC91/93/94 events via `ToneShaper.to_cc_events`.
By default CC91 controls reverb send, **93 controls delay**, and **94 controls chorus**.

## FX Envelope Example

```yaml
fx_envelope:
  0.0:
    cc: 91
    start_val: 0
    end_val: 100
    duration_ql: 4.0
    shape: lin
  4.0:
    cc: 91
    start_val: 100
    end_val: 20
    duration_ql: 2.0
    shape: exp
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
