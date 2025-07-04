# Live Tips

### Realtime Options

Common CLI options:

- `--late-humanize` shifts note timing a few milliseconds right before playback.
- `--rhythm-schema` prepends a rhythm style token when sampling transformer bass.
- `--normalize-lufs` normalises rendered audio to the given loudness target.
- `normalize_wav` can also infer targets per section using
  `{'verse': -16, 'chorus': -12}`.
- `--kick-leak-jitter` adjusts hi-hat velocity when kick leakage is detected.
- `--expr-curve` chooses the curve for expression CC generation.
- `--buffer-ahead` and `--parallel-bars` control the pre-generation buffer for
  live mode. Increase them if generation is slow.

Example usage enabling both options:

```bash
modcompose live model.pkl --late-humanize 6 --kick-leak-jitter 3
```
- ToneShaper selects amp presets using both intensity and average note
  velocity, then emits CC31 at the start of each part. Use it automatically at
  the end of `BassGenerator.compose()`:

  ```python
  from utilities.tone_shaper import ToneShaper

  shaper = ToneShaper()
  preset = shaper.choose_preset(avg_vel, "medium")
  part.extra_cc.extend(shaper.to_cc_events(preset, 0.0))
  ```

Run with automatic tone shaping:

```bash
modcompose render spec.yml --tone-auto
```

To emit CC11 and aftertouch for dynamic playback enable the flags programmatically:

```python
from utilities import humanizer

humanizer.set_cc_flags(True, True)
```

See [tone.md](tone.md) for details.
For velocity curves and jitter options see the [Humanizer Reference](humanizer.md).
