# Changelog

## Migration 0.9 → 1.0
- `ToneShaper.choose_preset` now requires keyword arguments: `amp_hint`, `intensity`, `avg_velocity`.
- `ToneShaper.to_cc_events` expects `amp_name` and returns a set of tuples.
- `merge_cc_events` accepts any `Iterable` but pass `set(base)` when combining lists.
- `ToneShaper.choose_preset` falls back to `<amp_hint>_default` when `amp_hint`
  isn't found.

