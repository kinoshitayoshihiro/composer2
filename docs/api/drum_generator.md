# DrumGenerator API Reference

## Class: DrumGenerator
A generator that creates drum parts based on emotion and intensity mappings. It inherits from `BasePartGenerator` and supports advanced timing and velocity features.

### Methods

#### `__init__(self, *, global_settings=None, default_instrument=None, global_tempo=None, global_time_signature=None, global_key_signature_tonic=None, global_key_signature_mode=None, main_cfg=None, drum_map=None, tempo_map=None, **kwargs)`
```python
class DrumGenerator(BasePartGenerator):
    def __init__(
        self,
        *,
        global_settings=None,
        default_instrument=None,
        global_tempo=None,
        global_time_signature=None,
        global_key_signature_tonic=None,
        global_key_signature_mode=None,
        main_cfg=None,
        drum_map=None,
        tempo_map=None,
        **kwargs,
    )
```
Initializes the drum generator with global tempo information, velocity smoothing and pattern libraries.

**Parameters**
- `global_settings` (dict): project‑wide defaults and feature flags.
- `main_cfg` (dict): main configuration loaded from YAML.
- `tempo_map` (TempoMap | None): optional tempo map instance.
- `global_tempo` (int | None): default BPM when no tempo map is provided.

**Returns**: `None`

Example:
```python
from generator.drum_generator import DrumGenerator
from music21 import stream

dg = DrumGenerator(main_cfg=my_cfg, global_settings=cli_args)
part = dg.compose(section_data=my_section)
```

### Global Settings

- `walk_after_ema` (bool): when `use_velocity_ema` is enabled, controls whether
  the random walk is applied after smoothing (`True`) or before (`False`).
- `export_random_walk_cc` (bool): if `True`, writes the random walk value to
  controller **20** once per bar for debugging.
- `random_walk_step` (int): base step range of the velocity random walk.
- `bar_start_abs_offset` is forwarded to `AccentMapper.begin_bar` so debug CC timings align.

### Consonant Sync Settings

| Key | Description | Default |
|-----|-------------|---------|
| `consonant_sync_mode` | `'bar'` or `'note'` | `'bar'` |
| `consonant_sync.note_radius_ms` | search radius around a hit when aligning in note mode | `30.0` |
| `consonant_sync.velocity_boost` | velocity increase applied when a hit is shifted (use `return_vel=True` with `align_to_consonant` to get the boost) | `6` |

#### `compose(self, *, section_data: Optional[Dict[str, Any]] = None, overrides_root: Optional[Any] = None, groove_profile_path: Optional[str] = None, next_section_data: Optional[Dict[str, Any]] = None, part_specific_humanize_params: Optional[Dict[str, Any]] = None, shared_tracks: Dict[str, Any] | None = None) -> stream.Part`
```python
def compose(
    self,
    *,
    section_data: Optional[Dict[str, Any]] = None,
    overrides_root: Optional[Any] = None,
    groove_profile_path: Optional[str] = None,
    next_section_data: Optional[Dict[str, Any]] = None,
    part_specific_humanize_params: Optional[Dict[str, Any]] = None,
    shared_tracks: Dict[str, Any] | None = None,
) -> stream.Part
```
Generate a drum part for a section. Applies emotional mapping and optional overrides.

**Parameters**
- `section_data` (dict | None): metadata describing the musical section.
- `overrides_root` (Any | None): optional override model.

**Returns**: `music21.stream.Part` – rendered drum part.

#### `get_kick_offsets(self) -> List[float]`
```python
def get_kick_offsets(self) -> List[float]
```
Return a list of absolute offsets (in beats) where kick drums occur.

**Returns**: `list[float]`

#### `get_fill_offsets(self) -> List[float]`
```python
def get_fill_offsets(self) -> List[float]
```
Return positions of inserted fills.

**Returns**: `list[float]`

#### `_apply_pattern(self, part: stream.Part, events: List[Dict[str, Any]], bar_start_abs_offset: float, current_bar_actual_len_ql: float, pattern_base_velocity: int, swing_type: str, swing_ratio: float, current_pattern_ts: meter.TimeSignature, drum_block_params: Dict[str, Any], velocity_scale: float = 1.0, velocity_curve: List[float] | None = None, legato: bool = False) -> None`
```python
def _apply_pattern(
    self,
    part: stream.Part,
    events: List[Dict[str, Any]],
    bar_start_abs_offset: float,
    current_bar_actual_len_ql: float,
    pattern_base_velocity: int,
    swing_type: str,
    swing_ratio: float,
    current_pattern_ts: meter.TimeSignature,
    drum_block_params: Dict[str, Any],
    velocity_scale: float = 1.0,
    velocity_curve: List[float] | None = None,
    legato: bool = False,
) -> None
```
Insert a list of drum events into a music21 part. Supports articulations such as drag, ruff and flam.

Example:
```python
part = stream.Part()
pattern = [{"offset": 0.0, "instrument": "kick"}, {"offset": 2.0, "instrument": "snare"}]
dg._apply_pattern(part, pattern, 0.0, 4.0, 80, "eighth", 0.5, meter.TimeSignature("4/4"), {}, 1.0, [1.0])
```

#### `_make_hit(self, name: str, vel: int, ql: float, ev_def: Optional[Dict[str, Any]] = None) -> Optional[note.Note]`
```python
def _make_hit(
    self,
    name: str,
    vel: int,
    ql: float,
    ev_def: Optional[Dict[str, Any]] = None,
) -> Optional[note.Note]
```
Return a single drum hit as a `music21.note.Note`.

Example:
```python
note_obj = dg._make_hit("snare", 100, 0.25)
```

#### `_insert_grace_chain(self, part: stream.Part, offset: float, midi_pitch: int, velocity: int, n_hits: int = 2, *, spread_ms: float = 25.0, velocity_curve: str | Sequence[float] | None = None, humanize: bool | str | dict | None = None, tempo_bpm: float | None = None) -> None`
```python
def _insert_grace_chain(
    self,
    part: stream.Part,
    offset: float,
    midi_pitch: int,
    velocity: int,
    n_hits: int = 2,
    *,
    spread_ms: float = 25.0,
    velocity_curve: str | Sequence[float] | None = None,
    humanize: bool | str | dict | None = None,
    tempo_bpm: float | None = None,
) -> None
```
Insert multiple grace notes leading into a main hit.

Example:
```python
dg._insert_grace_chain(part, 1.0, 38, 90, n_hits=3)
```

### Tom DSL Fill Patterns

`FillInserter` supports a compact DSL for tom fills when `pattern_type` is set to `"tom_dsl_fill"`.
Tokens like `T1`, `T2`, `T3`, `K` and `S` trigger drum hits, `+` extends the previous hit by a 16th note and `.` inserts a rest. Velocity modifiers such as `>1.2` scale the next hit, groups may repeat using `( ... )xN` (or default to one time when `xN` is omitted) and `@N` jumps to an absolute 16th offset.

Example YAML pattern:

```yaml
tom_run_short:
  description: "1 小節前半にタム回し"
  pattern_type: "tom_dsl_fill"
  length_beats: 1.0
  drum_base_velocity: 88
  pattern: |
    (T1 T2 T3 S)
```


