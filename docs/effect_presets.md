# Effect Preset Format

Effect presets map a name to an impulse response file and optional CC values.

```yaml
concert_hall:
  ir_file: "irs/hall.wav"
  cc:
    91: 80
    93: 20
```

Load presets with `EffectPresetLoader.load()` and retrieve them using `get()`.

```python
from utilities.effect_preset_loader import EffectPresetLoader
EffectPresetLoader.load("data/strings_fx.yml")
conf = EffectPresetLoader.get("concert_hall")
```

## Strings IR rendering

Use ``modcompose ir-render`` to apply impulse responses to strings sections:

```bash
modcompose ir-render strings.mid irs/hall.wav --part strings -o rendered.wav
```

Example of a multi-band EQ preset in JSON format:

```json
{
  "strings_eq": {
    "ir_file": "irs/strings_eq.wav",
    "cc": {
      "11": 80
    },
    "eq": {"low": -3, "mid": 2, "high": 1}
  }
}
```
