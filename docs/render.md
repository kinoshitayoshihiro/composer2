# Rendering Audio

Use `scripts/render_audio.py` to process breaths in a WAV file.

```bash
python scripts/render_audio.py input.wav -o output.wav \
  --breath-mode remove --hop-ms 5 --thr-off -25 --atten-gain -20 \
  --percentile 90 --onnx model.onnx --log-level info --dry-run
```

CLI options override YAML settings, which in turn override built-in defaults.

## Config Keys

| key | description | default |
| --- | ----------- | ------- |
| `breath_mode` | keep / attenuate / remove | `keep` |
| `attenuate_gain_db` | gain applied in attenuate mode | `-15` |
| `crossfade_ms` | remove mode crossfade length | `50` |
| `hop_ms` | analysis hop size | `10` |
| `thr_offset_db` | energy threshold offset | `-30` |
| `breath_threshold_offset_db` | deprecated alias for `thr_offset_db` | |
| `energy_percentile` | percentile for threshold | `95` |
| `log_level` | logging level | `WARN` |


CLI flag `--thr-off` remains supported as a deprecated alias for `thr_offset_db`.
Both `thr_offset_db` and the older `breath_threshold_offset_db` keys are accepted
in YAML configuration files.
