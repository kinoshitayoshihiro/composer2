# Amp & IR Guide

The tone system reads presets from `data/amp_presets.yml`.

```yaml
presets:
  clean: 20
  crunch: 50
ir:
  clean: "irs/blackface-clean.wav"
levels:
  clean: {reverb: 40, chorus: 20}
rules:
  - if: "intensity=='high'"
    preset: crunch
```

## CC Mapping

| Parameter | CC# |
|-----------|----|
| Amp type  | 31 |
| Reverb    | 91 |
| Chorus    | 93 |
| Delay     | 94 |

Impulse responses are looked up relative to the project root under `irs/`.
