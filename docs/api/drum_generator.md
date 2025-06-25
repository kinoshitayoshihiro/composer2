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


