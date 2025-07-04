# FX Rendering Guide

ToneShaper reads amp and effect presets from `data/amp_presets.yml`.
A minimal file looks like:

```yaml
presets:
  clean: 20
  crunch: 50
levels:
  clean: {reverb: 40, chorus: 20, delay: 10}
ir:
  clean: "irs/blackface-clean.wav"
```

Place impulse responses under the `irs/` directory relative to the
project root. The CLI renders audio with:

```bash
modcompose fx render demo.mid --preset clean --out demo.wav
```

Impulse responses are licensed under CC-BY 4.0.
