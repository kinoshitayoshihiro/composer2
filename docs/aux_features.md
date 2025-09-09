# Groove Sampler Auxiliary Features

Version 1.1 lets the n-gram model specialise by section, heatmap bin and intensity.
Provide a JSON map when training:

```json
{
  "loop_01.mid": {"section": "verse", "heat_bin": 3, "intensity": "mid"},
  "loop_02.mid": {"section": "chorus", "heat_bin": 7, "intensity": "high"}
}
```

Missing keys raise an error. During sampling you may supply a partial condition:

```bash
modcompose groove sample model.pkl --cond '{"section":"chorus"}' > out.mid
```

The sampler first checks the full tuple `(section, heat_bin, intensity)`,
then falls back to a wildcard intensity and finally the global distribution.

Inspect a saved model with:

```bash
modcompose groove info model.pkl --json --stats
```

The `--auto-aux` option of `modcompose loops scan` estimates metadata
automatically. Intensity is `low` for mean velocity ≤60, `mid` for 61–100
and `high` otherwise. The heat bin corresponds to the step with the
highest number of hits modulo 16.
