import logging, random, json, copy
import math
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set, Sequence
from music21 import (
    stream,
    note,
    pitch,
    meter,
    instrument as m21instrument,
    duration as m21duration,
    volume as m21volume,
    tie,
    converter,
)

from .base_part_generator import BasePartGenerator
from utilities.core_music_utils import (
    get_time_signature_object,
    MIN_NOTE_DURATION_QL,
)
from utilities.onset_heatmap import build_heatmap, RESOLUTION, load_heatmap
from utilities import humanizer
from utilities import groove_sampler
from utilities import fill_dsl
from utilities.humanizer import apply_humanization_to_element
from utilities.safe_get import safe_get
from utilities.drum_map_registry import (
    DRUM_MAP,
    get_drum_map,
    GM_DRUM_MAP,
    MISSING_DRUM_MAP_FALLBACK,
)
from utilities.drum_map import GENERAL_MIDI_MAP
from utilities.timing_utils import TimingBlend, interp_curve, _combine_timing
from utilities.tempo_utils import get_tempo_at_beat, load_tempo_curve
from utilities.velocity_smoother import EMASmoother
from utilities.peak_synchroniser import PeakSynchroniser


logger = logging.getLogger("modular_composer.drum_generator")

# Hat suppression: omit hi-hat hits when relative vocal activity exceeds this
# threshold (0-1 scale based on heatmap weight).
HAT_SUPPRESSION_THRESHOLD = 0.6

# Default grace-note timing range in milliseconds. ``MIN_GRACE_MS`` and
# ``MAX_GRACE_MS`` are used to clamp the ``spread_ms`` parameter for drag/ruff
# articulations so extremely short or long grace windows are avoided.
MIN_GRACE_MS = 5.0
MAX_GRACE_MS = 60.0

# Emotion/Intensity to drum style LUT
EMOTION_INTENSITY_LUT = {
    ("soft_reflective", "low"): "brush_light_loop",
    ("soft_reflective", "high"): "brush_build_loop",
    ("super_drive", "low"): "rock_backbeat",
    ("super_drive", "high"): "rock_drive_loop",
}

DRUM_ALIAS: Dict[str, str] = {
    "hh": "hh",
    "hat_closed": "hat_closed",
    "ohh": "ohh",
    "shaker_soft": "shaker_soft",
    "chimes": "chimes",
    "ride_cymbal_swell": "ride_cymbal_swell",
    "crash_cymbal_soft_swell": "crash_cymbal_soft_swell",
    "ride": "ride_cymbal_swell",
    "ride_bell": "ride_bell",
    "splash": "splash",
    "crash_choke": "crash_choke",
    "edge": "hh_edge",
    "pedal": "hh_pedal",
}
GHOST_ALIAS: Dict[str, str] = {"ghost_snare": "snare", "gs": "snare"}
# Map stick articulations to their brush counterparts
BRUSH_MAP: Dict[str, str] = {"kick": "brush_kick", "snare": "snare_brush"}


class HoldTie(tie.Tie):
    """Helper tie object with custom ``tieType`` attribute used in tests."""

    __slots__ = ("tieType",)

    def __init__(self) -> None:
        super().__init__("continue")
        self.tieType = "hold"


def resolve_velocity_curve(curve_spec: Any) -> List[float]:
    """Return list of velocity multipliers from specification.

    Parameters
    ----------
    curve_spec : Any
        Either a list of numbers or a named preset string.

    Returns
    -------
    List[float]
        Multipliers for each subdivision. Defaults to ``[1.0]``.
    """
    if not curve_spec:
        return [1.0]

    if isinstance(curve_spec, list):
        curve = [float(v) for v in curve_spec if isinstance(v, (int, float))]
        return curve or [1.0]

    if isinstance(curve_spec, str):
        spec = curve_spec.lower().strip()
        presets = {
            "crescendo": [0.8, 0.9, 1.0, 1.1, 1.2],
            "decrescendo": [1.2, 1.1, 1.0, 0.9, 0.8],
            "swell": [0.8, 1.0, 1.2, 1.0, 0.8],
            "flat": [1.0],
        }
        if spec in presets:
            return presets[spec]
        try:
            return [float(v) for v in spec.replace(",", " ").split()]
        except Exception:
            return [1.0]

    return [1.0]


class AccentMapper:
    """Map accent strength and ghost-hat density using vocal heatmap."""

    def __init__(
        self,
        threshold: float = 0.6,
        ghost_density_range: tuple[float, float] = (0.3, 0.8),
        rng: random.Random | None = None,
    ) -> None:
        self.threshold = threshold
        self.ghost_density_range = ghost_density_range
        self.rng = rng or random.Random()
        self._walk = 0
        self._prev: int | None = None

    def begin_bar(self, walk_range: int = 8) -> None:
        step = self.rng.randint(-walk_range, walk_range)
        self._walk = max(-127, min(127, self._walk + step))

    def accent(self, rel: float, velocity: int, *, step: bool = True) -> int:
        """Return velocity after applying accent and random walk."""
        return self.get_velocity(rel, velocity, apply_walk=step)

    def get_velocity(
        self,
        rel: float,
        base_velocity: int,
        *,
        apply_walk: bool = True,
        clamp: tuple[int, int] = (1, 127),
    ) -> int:
        if rel >= self.threshold:
            vel = int(base_velocity * 1.2)
        else:
            vel = base_velocity
        if apply_walk:
            vel += self._walk
        low, high = clamp
        vel = max(low, min(high, vel))
        if self._prev is not None:
            vel = int(round((self._prev + vel) / 2))
        self._prev = vel
        return vel

    def ghost_density(self, rel: float) -> float:
        low, high = self.ghost_density_range
        return high if rel < self.threshold else low


class FillInserter:
    """Insert drum fills at section boundaries."""

    def __init__(
        self,
        pattern_lib: Dict[str, Any],
        rng: random.Random | None = None,
        base_velocity: int = 80,
    ) -> None:
        self.pattern_lib = pattern_lib
        self.rng = rng or random.Random()
        self.drum_map: Dict[str, tuple[str, int]] = {}
        self.base_velocity = base_velocity

    def insert(
        self,
        part: stream.Part,
        section_data: Dict[str, Any],
        fill_key: Optional[str] = None,
    ) -> None:
        key = fill_key or section_data.get("drum_fill_at_end")
        if not key:
            return
        fill_def = self.pattern_lib.get(key)
        if fill_def is None:
            logger.warning("FillInserter.insert: fill pattern '%s' not found", key)
            return
        template = fill_def.get("template")
        if isinstance(template, list):
            template = self.rng.choice(template)
        if template is not None:
            events = fill_dsl.parse(
                str(template),
                fill_def.get("length_beats", 1.0),
                fill_def.get("velocity_factor", 1.0),
            )
        else:
            events = fill_def.get("pattern", [])
        if not events:
            return
        velocity_curve = fill_def.get("velocity_curve")
        if velocity_curve:
            curve = [float(v) for v in velocity_curve]
            for i, ev in enumerate(events):
                scale = curve[min(i, len(curve) - 1)]
                ev["velocity_factor"] = ev.get("velocity_factor", 1.0) * scale
        legato_mode = fill_def.get("mode") == "legato"
        base_vel = int(fill_def.get("base_velocity", self.base_velocity))
        start = (
            section_data.get("absolute_offset", 0.0)
            + section_data.get("q_length", 4.0)
            - 4.0
        )
        prev_note: note.Note | None = None
        prev_offset = 0.0
        for ev in events:
            inst = ev.get("instrument")
            if not inst:
                continue
            gm_name, midi_pitch = self.drum_map.get(inst, (None, None))
            if midi_pitch is None:
                logger.warning("Unknown drum label %s", inst)
                continue
            n = note.Note()
            n.pitch = pitch.Pitch(midi=midi_pitch)
            offset_val = float(ev.get("offset", 0.0))
            n.duration = m21duration.Duration(ev.get("duration", 0.25))
            n.volume = m21volume.Volume(
                velocity=int(base_vel * ev.get("velocity_factor", 1.0))
            )
            if legato_mode and prev_note is not None:
                prev_note.tie = HoldTie()
                prev_note.duration = m21duration.Duration(offset_val - prev_offset)
            part.insert(start + offset_val, n)
            prev_note = n
            prev_offset = offset_val


EMOTION_TO_BUCKET: Dict[str, str] = {  # (前回と同様)
    "quiet_pain_and_nascent_strength": "ballad_soft",
    "self_reproach_regret_deep_sadness": "ballad_soft",
    "memory_unresolved_feelings_silence": "ballad_soft",
    "reflective_transition_instrumental_passage": "ballad_soft",
    "deep_regret_gratitude_and_realization": "groove_mid",
    "supported_light_longing_for_rebirth": "groove_mid",
    "wavering_heart_gratitude_chosen_strength": "groove_mid",
    "hope_dawn_light_gentle_guidance": "groove_mid",
    "nature_memory_floating_sensation_forgiveness": "groove_mid",
    "acceptance_of_love_and_pain_hopeful_belief": "anthem_high",
    "trial_cry_prayer_unbreakable_heart": "anthem_high",
    "reaffirmed_strength_of_love_positive_determination": "anthem_high",
    "future_cooperation_our_path_final_resolve_and_liberation": "anthem_high",
    "default": "groove_mid",
    "neutral": "groove_mid",
}
BUCKET_INTENSITY_TO_STYLE: Dict[str, Dict[str, str]] = {  # (前回と同様)
    "ballad_soft": {
        "low": "no_drums_or_gentle_cymbal_swell",
        "medium_low": "ballad_soft_kick_snare_8th_hat",
        "medium": "ballad_soft_kick_snare_8th_hat",
        "medium_high": "rock_ballad_build_up_8th_hat",
        "high": "rock_ballad_build_up_8th_hat",
        "default": "ballad_soft_kick_snare_8th_hat",
    },
    "groove_mid": {
        "low": "ballad_soft_kick_snare_8th_hat",
        "medium_low": "rock_ballad_build_up_8th_hat",
        "medium": "rock_ballad_build_up_8th_hat",
        "medium_high": "anthem_rock_chorus_16th_hat",
        "high": "anthem_rock_chorus_16th_hat",
        "default": "rock_ballad_build_up_8th_hat",
    },
    "anthem_high": {
        "low": "rock_ballad_build_up_8th_hat",
        "medium_low": "anthem_rock_chorus_16th_hat",
        "medium": "anthem_rock_chorus_16th_hat",
        "medium_high": "anthem_rock_chorus_16th_hat",
        "high": "anthem_rock_chorus_16th_hat",
        "default": "anthem_rock_chorus_16th_hat",
    },
    "default_fallback_bucket": {
        "low": "no_drums",
        "medium_low": "default_drum_pattern",
        "medium": "default_drum_pattern",
        "medium_high": "default_drum_pattern",
        "high": "default_drum_pattern",
        "default": "default_drum_pattern",
    },
}


def _resolve_style(emotion: str, intensity: str, pattern_lib: Dict[str, Any]) -> str:
    # (前回と同様)
    bucket = EMOTION_TO_BUCKET.get(emotion.lower(), "default_fallback_bucket")
    style_map_for_bucket = BUCKET_INTENSITY_TO_STYLE.get(bucket)
    if not style_map_for_bucket:
        logger.error(
            f"DrumGen _resolve_style: CRITICAL - Bucket '{bucket}' not defined. Using 'default_drum_pattern'."
        )
        return "default_drum_pattern"
    resolved_style = style_map_for_bucket.get(intensity.lower())
    if not resolved_style:
        resolved_style = style_map_for_bucket.get("default", "default_drum_pattern")
    if resolved_style not in pattern_lib:
        logger.warning(
            f"DrumGen _resolve_style: Resolved style '{resolved_style}' (E:'{emotion}',I:'{intensity}') not in pattern_lib. Falling back to 'default_drum_pattern'."
        )
        if "default_drum_pattern" not in pattern_lib:
            logger.error(
                "DrumGen _resolve_style: CRITICAL - Fallback 'default_drum_pattern' also not in pattern_lib. Returning 'no_drums'."
            )
            return "no_drums"
        return "default_drum_pattern"
    return resolved_style


def extract_tempo_map_from_midi(vocal_midi_path: str) -> List[Tuple[float, float]]:
    # 提案通りの実装
    tempo_map = []
    try:
        midi_stream = converter.parse(vocal_midi_path)
        for element in midi_stream.flatten().notes:
            if isinstance(element, note.Note):
                tempo_qn = element.quarterLength
                if element.duration and element.duration.quarterLength > 0:
                    tempo_map.append(
                        (element.offset, tempo_qn / element.duration.quarterLength)
                    )
    except Exception as e:
        logger.error(f"Error extracting tempo map from MIDI: {e}")
    return tempo_map


def load_heatmap_data(heatmap_path: Optional[str]) -> Dict[int, int]:
    """ヒートマップデータをJSONファイルから読み込み、{grid_index: count} の辞書を返す。"""
    if not heatmap_path or not Path(heatmap_path).exists():
        logger.warning(f"Heatmap not found at '{heatmap_path}'. Using empty heatmap.")
        return {}
    try:
        with open(heatmap_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # JSONが [{"grid_index": 0, "count": 99}, ...] の形式であると仮定
            heatmap_dict = {item["grid_index"]: item["count"] for item in data}
            logger.info(
                f"Loaded heatmap data from {heatmap_path}: {len(heatmap_dict)} entries."
            )
            return heatmap_dict
    except Exception as e:
        logger.error(f"Error loading heatmap data: {e}")
        return {}




# DrumGenerator例
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
        **kwargs,
    ):
        self.main_cfg = main_cfg
        self.drum_map = drum_map or GENERAL_MIDI_MAP
        super().__init__(
            global_settings=global_settings,
            default_instrument=default_instrument,
            global_tempo=global_tempo,
            global_time_signature=global_time_signature,
            global_key_signature_tonic=global_key_signature_tonic,
            global_key_signature_mode=global_key_signature_mode,
            **kwargs,
        )
        self.overrides = None
        # ここに他の初期化処理をまとめて書く
        self.logger = logging.getLogger("modular_composer.drum_generator")
        self.part_parameters = kwargs.get("part_parameters", {})
        self.kick_offsets: List[float] = []
        # Track fill offsets along with fade width for each fill
        self.fill_offsets: List[Tuple[float, float]] = []
        global_cfg = self.main_cfg.get("global_settings", {}) if self.main_cfg else {}
        self.fade_beats_default = float(global_cfg.get("fill_fade_beats", 2.0))
        self.strict_drum_map = bool(
            (global_settings or {}).get("strict_drum_map", False)
        )
        self.drum_map_name = (global_settings or {}).get("drum_map", "gm")
        self.drum_map = get_drum_map(self.drum_map_name)
        # Simplified mapping to MIDI note numbers for internal use
        self.gm_pitch_map: Dict[str, int] = {}
        for label, (gm_name, midi) in self.drum_map.items():
            self.gm_pitch_map[label] = midi
            self.gm_pitch_map[gm_name] = midi
        self._warned_missing_drum_map: set[str] = set()
        # もし、この後に独自の初期化処理があれば、ここに残してください。
        # 必須のデフォルトパターンが不足している場合に補充
        self._add_internal_default_patterns()

        curve_path = (
            (global_settings or {}).get("tempo_curve_path")
            or (global_settings or {}).get("tempo_curve_json")
            or (self.main_cfg.get("paths", {}).get("tempo_curve_path") if self.main_cfg else None)
            or (self.main_cfg.get("paths", {}).get("tempo_curve_json") if self.main_cfg else None)
        )
        self.tempo_curve: list[dict] | None = None
        if curve_path:
            p = Path(curve_path).expanduser()
            if p.exists():
                try:
                    self.tempo_curve = load_tempo_curve(p)
                except Exception as e:  # pragma: no cover - optional
                    logger.warning(f"Failed to load tempo curve from {p}: {e}")

        self.current_bpm = self.main_cfg.get("tempo", 120)
        self.use_velocity_ema = bool(
            (global_settings or {}).get("use_velocity_ema", False)
        )
        self._vel_smoother = EMASmoother(window=16)

        sync_cfg = global_cfg.get("consonant_sync", {})
        self.consonant_sync_cfg = {
            "lag_ms": float(sync_cfg.get("lag_ms", 10.0)),
            "min_distance_beats": float(sync_cfg.get("min_distance_beats", 0.25)),
            "sustain_threshold_ms": float(sync_cfg.get("sustain_threshold_ms", 120.0)),
        }
        self.consonant_peaks: List[float] | None = None

        # Use path settings from main_cfg, preferring an explicit MIDI file path
        self.vocal_midi_path = (
            self.main_cfg.get("paths", {}).get("vocal_midi_path_for_drums")
            or self.main_cfg.get("vocal_midi_path_for_drums")
            or self.main_cfg.get("paths", {}).get("vocal_note_data_path")
        )
        self.vocal_end_times: List[float] = self._load_vocal_end_times(
            self.vocal_midi_path
        )

        heatmap_json_path = self.main_cfg.get("heatmap_json_path_for_drums")
        if not heatmap_json_path:
            heatmap_json_path = self.main_cfg.get("paths", {}).get("vocal_heatmap_path")
        if not heatmap_json_path:
            heatmap_json_path = str(Path("data/heatmap.json").resolve())

        heatmap_json_path = str(Path(heatmap_json_path).expanduser().resolve())
        self.heatmap = load_heatmap(heatmap_json_path)
        self.max_heatmap_value = max(self.heatmap.values()) if self.heatmap else 0
        self.heatmap_resolution = self.main_cfg.get("heatmap_resolution", RESOLUTION)
        self.heatmap_threshold = self.main_cfg.get("heatmap_threshold", 1)
        # Velocity below this value triggers use of the HH edge articulation
        self.hh_edge_threshold = int(self.main_cfg.get("hh_edge_threshold", 50))
        self.rng = random.Random()
        if self.main_cfg.get("rng_seed") is not None:
            self.rng.seed(self.main_cfg["rng_seed"])

        self.accent_mapper = AccentMapper(
            threshold=self.main_cfg.get("accent_threshold", 0.6),
            ghost_density_range=tuple(
                self.main_cfg.get("ghost_density_range", [0.3, 0.8])
            ),
            rng=self.rng,
        )
        self.ghost_hat_on_offbeat = self.main_cfg.get("ghost_hat_on_offbeat", True)

        # Drum ブラシ（テストでは drum_brush を参照）
        self.drum_brush = bool(
            safe_get(
                global_cfg, "drum_brush", default=self.main_cfg.get("drum_brush", False)
            )
            or (self.overrides and self.overrides.drum_brush)
        )

        # ランダムウォークのステップ幅（テストで random_walk_step を使う場合）
        self.random_walk_step = int(self.main_cfg.get("random_walk_step", 0))

        # apply groove pretty
        global_cfg = self.main_cfg.get("global_settings", {})
        self.random_walk_step = int(
            global_cfg.get("random_walk_step", self.random_walk_step)
        )
        self.groove_profile_path = global_cfg.get("groove_profile_path")
        self.groove_strength = float(global_cfg.get("groove_strength", 1.0))
        self.groove_profile = {}
        if self.groove_profile_path:
            try:
                with open(self.groove_profile_path, "r", encoding="utf-8") as f:
                    self.groove_profile = json.load(f)
            except Exception as e:
                logger.warning(
                    f"Failed to load groove profile from {self.groove_profile_path}: {e}"
                )

        groove_dir = global_cfg.get("groove_midi_dir")
        groove_ngram = int(global_cfg.get("groove_ngram", 2))
        groove_resolution = int(global_cfg.get("groove_resolution", 16))
        self.groove_resolution = groove_resolution
        self.groove_model = {}
        if groove_dir:
            try:
                self.groove_model = groove_sampler.load_grooves(
                    Path(groove_dir), n=groove_ngram, resolution=groove_resolution
                )
            except Exception as e:  # pragma: no cover - optional feature
                logger.warning(f"Failed to load grooves from {groove_dir}: {e}")
        self._groove_history: List[str] = []

        self.push_pull_map = global_cfg.get("push_pull_curve", {})
        self.current_push_pull_curve = None

        self.push_pull_max_ms = float(global_cfg.get("push_pull_max_ms", 80.0))
        self.open_hat_choke_prob = float(global_cfg.get("open_hat_choke_prob", 0.35))
        self.open_hat_choke_min_ms = float(global_cfg.get("open_hat_choke_min_ms", 160.0))
        self.open_hat_choke_max_ms = float(global_cfg.get("open_hat_choke_max_ms", 280.0))
        vr = global_cfg.get("velocity_range", (0.9, 1.1))
        if (
            isinstance(vr, (list, tuple))
            and len(vr) == 2
            and all(isinstance(v, (int, float)) for v in vr)
        ):
            self.velocity_range = (float(vr[0]), float(vr[1]))
        else:
            self.velocity_range = (0.9, 1.1)
        
        # 楽器設定
        part_default_cfg = self.main_cfg.get("default_part_parameters", {}).get(
            self.part_name, {}
        )
        instrument_name = part_default_cfg.get("instrument", "DrumSet")
        try:
            self.default_instrument = m21instrument.Percussion()
            if hasattr(self.default_instrument, "midiChannel"):
                self.default_instrument.midiChannel = 9
        except:
            self.default_instrument = m21instrument.Percussion()

        # (初期化ロジックは前回と同様)
        self.raw_pattern_lib = (
            copy.deepcopy(self.part_parameters)
            if self.part_parameters is not None
            else {}
        )
        self.pattern_lib_cache: Dict[str, Dict[str, Any]] = {}
        logger.info(
            f"DrumGen __init__: Initialized with {len(self.raw_pattern_lib)} raw drum patterns."
        )
        self.fill_inserter = FillInserter(self.raw_pattern_lib)
        self.fill_inserter.drum_map = self.drum_map
        core_defaults = {
            "default_drum_pattern": {
                "description": "Default fallback pattern",
                "pattern": [
                    {
                        "offset": 0.0,
                        "duration": 0.125,
                        "instrument": "kick",
                        "velocity_factor": 1.0,
                    },
                    {
                        "offset": 2.0,
                        "duration": 0.125,
                        "instrument": "snare",
                        "velocity_factor": 0.9,
                    },
                ],
                "time_signature": "4/4",
                "swing": 0.5,
                "length_beats": 4.0,
                "fill_ins": {},
                "velocity_base": 80,
            },
            "no_drums": {
                "description": "Silence",
                "pattern": [],
                "time_signature": "4/4",
                "swing": 0.5,
                "length_beats": 4.0,
                "fill_ins": {},
                "velocity_base": 0,
            },
            "no_drums_or_gentle_cymbal_swell": {
                "description": "Placeholder: Gentle cymbal swell or silence",
                "pattern": [],
                "velocity_base": 50,
            },
            "ballad_soft_kick_snare_8th_hat": {
                "description": "Placeholder: Soft ballad beat",
                "pattern": [],
                "velocity_base": 70,
            },
            "rock_ballad_build_up_8th_hat": {
                "description": "Placeholder: Rock ballad build-up",
                "pattern": [],
                "velocity_base": 85,
            },
            "anthem_rock_chorus_16th_hat": {
                "description": "Placeholder: Anthem rock chorus",
                "pattern": [],
                "velocity_base": 100,
            },
            "no_drums_or_sparse_cymbal": {
                "description": "Placeholder: Sparse cymbal or silence",
                "pattern": [],
                "velocity_base": 40,
            },
            "no_drums_or_sparse_chimes": {
                "description": "Placeholder: Sparse chimes or silence",
                "pattern": [],
                "velocity_base": 45,
            },
        }
        for k, v_def_template in core_defaults.items():
            if k not in self.raw_pattern_lib:
                placeholder_def = {
                    "description": v_def_template.get(
                        "description", f"Placeholder for '{k}'."
                    ),
                    "pattern": v_def_template.get("pattern", []),
                    "time_signature": v_def_template.get("time_signature", "4/4"),
                    "swing": v_def_template.get("swing", 0.5),
                    "length_beats": v_def_template.get("length_beats", 4.0),
                    "fill_ins": v_def_template.get("fill_ins", {}),
                    "velocity_base": v_def_template.get("velocity_base", 70),
                }
                self.raw_pattern_lib[k] = placeholder_def
                logger.info(
                    f"DrumGen __init__: Added/updated placeholder for style '{k}'."
                )
        all_referenced_styles_in_luts: Set[str] = set()
        for bucket_styles in BUCKET_INTENSITY_TO_STYLE.values():
            all_referenced_styles_in_luts.update(bucket_styles.values())
        for style_key in all_referenced_styles_in_luts:
            if style_key not in self.raw_pattern_lib:
                self.raw_pattern_lib[style_key] = {
                    "description": f"Auto-added placeholder for undefined style '{style_key}'.",
                    "pattern": [],
                    "time_signature": "4/4",
                    "swing": 0.5,
                    "length_beats": 4.0,
                    "fill_ins": {},
                    "velocity_base": 70,
                }
                logger.info(
                    f"DrumGen __init__: Added silent placeholder for undefined style '{style_key}' (from LUT)."
                )
        self.global_tempo = self.main_cfg.get("tempo", 120)
        self.global_time_signature_str = self.main_cfg.get("time_signature", "4/4")
        self.global_ts = get_time_signature_object(self.global_time_signature_str)
        if not self.global_ts:
            logger.warning(
                f"DrumGen __init__: Failed to parse global time_sig '{self.global_time_signature_str}'. Defaulting to 4/4."
            )
            self.global_ts = meter.TimeSignature("4/4")
        self.instrument = m21instrument.Percussion()
        if hasattr(self.instrument, "midiChannel"):
            self.instrument.midiChannel = 9

        # rhythm_library.yml 内の drum_patterns をロードして raw_pattern_lib にマージ
        lib_path = self.main_cfg.get("paths", {}).get(
            "rhythm_library_path", "data/rhythm_library.yml"
        )
        lib_path = str(Path(lib_path).expanduser())
        try:
            with open(lib_path, "r", encoding="utf-8") as fh:
                library_data = yaml.safe_load(fh) or {}
                if isinstance(library_data, dict):
                    self.raw_pattern_lib.update(library_data.get("drum_patterns", {}))
        except FileNotFoundError:
            logger.warning(f"DrumGen __init__: rhythm library not found: {lib_path}")
        except Exception as exc:
            logger.warning(
                f"DrumGen __init__: failed to load rhythm library {lib_path}: {exc}"
            )

        # 最終的なパターン辞書を part_parameters に適用
        self.part_parameters = self.raw_pattern_lib

    def _current_bpm(self, abs_offset_ql: float) -> float:
        """Return BPM at ``abs_offset_ql`` using tempo curve if available."""
        if self.tempo_curve is not None:
            return get_tempo_at_beat(abs_offset_ql, self.tempo_curve)
        return self.global_tempo

    def _choose_pattern_key(
        self,
        emotion: str | None,
        intensity: str | None,
        musical_intent: Optional[Dict[str, Any]] = None,
    ) -> str:
        emo = (emotion or "default").lower()
        inten = (intensity or "medium").lower()
        bucket = EMOTION_TO_BUCKET.get(emo, "default_fallback_bucket")
        style_map = BUCKET_INTENSITY_TO_STYLE.get(bucket, {})
        base_key = style_map.get(inten) or style_map.get(
            "default", "default_drum_pattern"
        )
        if musical_intent and musical_intent.get("syncopation"):
            for k, v in self.raw_pattern_lib.items():
                if "offbeat" in v.get("tags", []):
                    return k
        return base_key

    def _get_effective_pattern_def(
        self, style_key: str, visited: Optional[Set[str]] = None
    ) -> Dict[str, Any]:
        # (前回と同様の継承解決ロジック)
        if visited is None:
            visited = set()
        if style_key in visited:
            logger.error(
                f"DrumGen: Circular inheritance for '{style_key}'. Returning 'default_drum_pattern'."
            )
            default_p_data = self.pattern_lib_cache.get(
                "default_drum_pattern"
            ) or self.raw_pattern_lib.get("default_drum_pattern", {})
            return copy.deepcopy(default_p_data if default_p_data else {"pattern": []})
        if style_key in self.pattern_lib_cache:
            return copy.deepcopy(self.pattern_lib_cache[style_key])
        pattern_def_original = self.raw_pattern_lib.get(style_key)
        if not pattern_def_original:
            logger.warning(
                f"DrumGen: Style key '{style_key}' not found. Falling back to 'default_drum_pattern'."
            )
            default_p = self.raw_pattern_lib.get("default_drum_pattern")
            if not default_p:
                logger.error(
                    "DrumGen: CRITICAL - 'default_drum_pattern' missing. Returning minimal empty."
                )
                return {
                    "description": "Minimal Empty (Critical Fallback)",
                    "pattern": [],
                    "time_signature": "4/4",
                    "swing": 0.5,
                    "length_beats": 4.0,
                    "fill_ins": {},
                    "velocity_base": 70,
                }
            self.pattern_lib_cache[style_key] = copy.deepcopy(default_p)
            return default_p
        pattern_def = copy.deepcopy(pattern_def_original)
        inherit_key = pattern_def.get("inherit")
        if inherit_key and isinstance(inherit_key, str):
            logger.debug(
                f"DrumGen: Pattern '{style_key}' inherits '{inherit_key}'. Resolving..."
            )
            visited.add(style_key)
            base_def = self._get_effective_pattern_def(inherit_key, visited)
            visited.remove(style_key)
            merged_def = base_def.copy()
            if "pattern" in pattern_def:
                merged_def["pattern"] = pattern_def["pattern"]
            base_fills = merged_def.get("fill_ins", {})
            current_fills = pattern_def.get("fill_ins", {})
            if isinstance(base_fills, dict) and isinstance(current_fills, dict):
                merged_fills = base_fills.copy()
                merged_fills.update(current_fills)
                merged_def["fill_ins"] = merged_fills
            elif current_fills is not None:
                merged_def["fill_ins"] = current_fills
            for key, value in pattern_def.items():
                if key not in ["inherit", "pattern", "fill_ins"]:
                    merged_def[key] = value
            pattern_def = merged_def
        pattern_def.setdefault("time_signature", self.global_time_signature_str)
        pattern_def.setdefault("swing", 0.5)
        pattern_def.setdefault(
            "length_beats",
            (
                get_time_signature_object(
                    pattern_def["time_signature"]
                ).barDuration.quarterLength
                if get_time_signature_object(pattern_def["time_signature"])
                else 4.0
            ),
        )
        pattern_def.setdefault("pattern", [])
        pattern_def.setdefault("fill_ins", {})
        pattern_def.setdefault("velocity_base", 80)
        pattern_def.setdefault("fill_patterns", [])
        pattern_def.setdefault("preferred_fill_positions", [])
        pattern_def["velocity_curve"] = resolve_velocity_curve(
            pattern_def.get("options", {}).get("velocity_curve")
        )
        self.pattern_lib_cache[style_key] = copy.deepcopy(pattern_def)
        return pattern_def

    def compose(
        self,
        *,
        section_data: Optional[Dict[str, Any]] = None,
        overrides_root: Optional[Any] = None,
        groove_profile_path: Optional[str] = None,
        next_section_data: Optional[Dict[str, Any]] = None,
        part_specific_humanize_params: Optional[Dict[str, Any]] = None,
        shared_tracks: Dict[str, Any] | None = None,
    ) -> stream.Part:
        """
        mode == "independent" : ボーカル熱マップ主導で全曲を一括生成
        mode == "chord"      : chordmap のセクション単位で生成
        共通APIを維持しつつ、必要なときだけ独自処理を挟む。
        """
        # Reset stateful tracking of fills each time compose is called so
        # consecutive invocations don't accumulate offsets.
        self.fill_offsets.clear()
        if getattr(self, "mode", "chord") == "independent":
            return self._render_whole_song()

        # Configuration for heatmap processing
        self.heatmap_resolution = (self.main_cfg or {}).get(
            "heatmap_resolution"
        ) or self.global_settings.get("heatmap_resolution", RESOLUTION)
        self.heatmap_threshold = (self.main_cfg or {}).get(
            "heatmap_threshold"
        ) or self.global_settings.get("heatmap_threshold", 0.5)

        if section_data and section_data.get("label") in {"Intro", "Outro"}:
            section_data.setdefault("part_params", {}).setdefault(self.part_name, {})[
                "rhythm_key"
            ] = "ride_only"
            section_data["part_params"][self.part_name][
                "final_style_key_for_render"
            ] = "ride_only"

        if section_data and section_data.get("expression_details"):
            expr = section_data["expression_details"]
            key = (expr.get("emotion_bucket"), expr.get("intensity"))
            mapped = EMOTION_INTENSITY_LUT.get(key)
            if mapped:
                section_data.setdefault("part_params", {}).setdefault(
                    self.part_name, {}
                )["rhythm_key"] = mapped

        emotion_key = None
        if section_data:
            emotion_key = (section_data.get("expression_details") or {}).get(
                "emotion_bucket"
            ) or section_data.get("musical_intent", {}).get("emotion")
        self.current_push_pull_curve = (
            self.push_pull_map.get(emotion_key) if emotion_key else None
        )

        part = super().compose(
            section_data=section_data,
            overrides_root=overrides_root,
            groove_profile_path=groove_profile_path or self.groove_profile_path,
            next_section_data=next_section_data,
            part_specific_humanize_params=part_specific_humanize_params,
            shared_tracks=shared_tracks,
        )

        if section_data:
            self.fill_inserter.insert(part, section_data)
        self._sync_hihat_with_vocals(part)
        return part

    def _render(
        self,
        blocks: Sequence[Dict[str, Any]],
        part: stream.Part,
        section_data: Optional[Dict[str, Any]] = None,
    ):
        ms_since_fill = 0
        bars_since_section_start = 0
        num_bars = (
            section_data.get("length_in_measures", len(blocks))
            if section_data
            else len(blocks)
        )
        total_beats = sum(
            b.get("humanized_duration_beats", b.get("q_length", 4.0)) for b in blocks
        )
        running_beats = 0.0
        for blk_idx, blk_data in enumerate(blocks):
            log_render_prefix = f"DrumGen.Render.Blk{blk_idx+1}"  # 1-indexed for logs
            intensity = blk_data.get("musical_intent", {}).get("emotion_intensity", 0.0)
            fill_threshold = self.main_cfg.get("fill_emotion_threshold", 0.8)
            if intensity >= fill_threshold and section_data is not None:
                peak_pos = running_beats / total_beats if total_beats > 0 else 0.0
                peak_idx = peak_pos * num_bars
                peak_bar = math.floor(peak_idx) + 1
                peak_bar = max(1, min(num_bars, peak_bar))
                section_data.setdefault("preferred_fill_positions", []).append(peak_bar)
            drums_params = blk_data.get("part_params", {}).get("drums", {})
            style_key = drums_params.get(
                "final_style_key_for_render", "default_drum_pattern"
            )
            style_def = self._get_effective_pattern_def(style_key)
            if not style_def:
                logger.error(
                    f"{log_render_prefix}: CRITICAL - No style_def for '{style_key}'. Skipping."
                )
                continue

            style_options = style_def.get("options", {})
            velocity_curve_list = resolve_velocity_curve(
                style_options.get("velocity_curve")
            )

            # --- base_vel の取得 (safe_get を使用) ---
            base_vel = safe_get(
                drums_params,
                "velocity",
                default=safe_get(
                    drums_params,
                    "drum_base_velocity",
                    default=safe_get(
                        style_def,
                        "velocity_base",
                        default=80,
                        cast_to=int,
                        log_name=f"{log_render_prefix}.VelStyleDef",
                    ),
                    cast_to=int,
                    log_name=f"{log_render_prefix}.VelDrumBaseParam",
                ),
                cast_to=int,
                log_name=f"{log_render_prefix}.VelParam",
            )
            base_vel = max(1, min(127, base_vel))
            # --- ここまで base_vel ---

            pat_events: List[Dict[str, Any]] = style_def.get("pattern", [])
            pat_ts_str = style_def.get("time_signature", self.global_time_signature_str)
            pat_ts = get_time_signature_object(pat_ts_str)
            if not pat_ts:
                pat_ts = self.global_ts

            pattern_unit_length_ql = safe_get(
                style_def,
                "length_beats",
                default=pat_ts.barDuration.quarterLength if pat_ts else 4.0,
                cast_to=float,
                log_name=f"{log_render_prefix}.PatternLen",
            )
            if pattern_unit_length_ql <= 0:
                logger.warning(
                    f"{log_render_prefix}: Pattern '{style_key}' invalid length {pattern_unit_length_ql}. Defaulting to 4.0"
                )
                pattern_unit_length_ql = 4.0

            swing_setting = style_def.get("swing", 0.5)
            swing_type = "eighth"
            swing_ratio_val = 0.5
            if isinstance(swing_setting, dict):
                swing_type = swing_setting.get("type", "eighth").lower()
                swing_ratio_val = safe_get(
                    swing_setting,
                    "ratio",
                    default=0.5,
                    cast_to=float,
                    log_name=f"{log_render_prefix}.SwingRatio",
                )
            elif isinstance(swing_setting, (float, int)):
                swing_ratio_val = float(swing_setting)

            fills = style_def.get("fill_ins", {})

            # --- オフセットとデュレーション (safe_get を使用) ---
            default_block_dur = (
                pattern_unit_length_ql if pattern_unit_length_ql > 0 else 4.0
            )
            offset_in_score = safe_get(
                blk_data,
                "humanized_offset_beats",
                default=safe_get(
                    blk_data,
                    "absolute_offset",
                    default=safe_get(
                        blk_data,
                        "offset",
                        default=0.0,
                        cast_to=float,
                        log_name=f"{log_render_prefix}.OffsetFallback3",
                    ),
                    cast_to=float,
                    log_name=f"{log_render_prefix}.OffsetFallback2",
                ),
                cast_to=float,
                log_name=f"{log_render_prefix}.HumOffset",
            )
            remaining_ql_in_block = safe_get(
                blk_data,
                "humanized_duration_beats",
                default=safe_get(
                    blk_data,
                    "q_length",
                    default=default_block_dur,
                    cast_to=float,
                    log_name=f"{log_render_prefix}.QLFallback",
                ),
                cast_to=float,
                log_name=f"{log_render_prefix}.HumDur",
            )
            if remaining_ql_in_block <= 0:
                logger.warning(
                    f"{log_render_prefix}: Non-positive duration {remaining_ql_in_block} (raw: {blk_data.get('humanized_duration_beats', blk_data.get('q_length'))}). Using {default_block_dur}ql."
                )
                remaining_ql_in_block = default_block_dur
            # --- ここまでオフセットとデュレーション ---

            if blk_data.get("is_first_in_section", False) and blk_idx > 0:
                ms_since_fill = 0
                bars_since_section_start = 0
            current_pos_within_block = 0.0
            while remaining_ql_in_block > MIN_NOTE_DURATION_QL / 8.0:
                self.accent_mapper.begin_bar(self.random_walk_step)
                # (フィルインロジック、パターンの適用は前回と同様、base_vel を _apply_pattern に渡す)
                current_pattern_iteration_ql = min(
                    pattern_unit_length_ql, remaining_ql_in_block
                )
                if current_pattern_iteration_ql < MIN_NOTE_DURATION_QL / 4.0:
                    break
                is_last_pattern_iteration_in_block = (
                    remaining_ql_in_block
                    <= pattern_unit_length_ql + (MIN_NOTE_DURATION_QL / 8.0)
                )
                pattern_to_use_for_iteration = pat_events
                fill_applied_this_iter = False
                fill_legato = False
                override_fill_key = drums_params.get(
                    "fill_override", drums_params.get("drum_fill_key_override")
                )
                if is_last_pattern_iteration_in_block and override_fill_key:
                    fill_def = self._get_effective_pattern_def(override_fill_key)
                    chosen_fill_pattern_list = fill_def.get("pattern", [])
                    if chosen_fill_pattern_list is not None:
                        pattern_to_use_for_iteration = chosen_fill_pattern_list
                        fill_legato = bool(fill_def.get("legato"))
                        fill_applied_this_iter = True
                        logger.debug(
                            f"{log_render_prefix}: Applied override fill '{override_fill_key}' for style '{style_key}'."
                        )
                    else:
                        logger.warning(
                            f"{log_render_prefix}: Override fill key '{override_fill_key}' not in fills for '{style_key}'."
                        )

                preferred_positions = [
                    int(p)
                    for p in style_def.get("preferred_fill_positions", [])
                    if isinstance(p, int)
                ]
                if section_data:
                    preferred_positions.extend(
                        int(p)
                        for p in section_data.get("preferred_fill_positions", [])
                        if isinstance(p, int)
                    )
                fill_keys = style_def.get("fill_patterns", [])
                at_section_end = (
                    blk_idx == len(blocks) - 1 and is_last_pattern_iteration_in_block
                )
                bar_number = bars_since_section_start + 1
                if (
                    not fill_applied_this_iter
                    and fill_keys
                    and (bar_number in preferred_positions or at_section_end)
                ):
                    candidates = [
                        fk
                        for fk in fill_keys
                        if self._get_effective_pattern_def(fk).get(
                            "length_beats", pattern_unit_length_ql
                        )
                        == pattern_unit_length_ql
                    ]
                    if candidates:
                        fill_key = self.rng.choice(candidates)
                        fill_def = self._get_effective_pattern_def(fill_key)
                        pattern_to_use_for_iteration = fill_def.get("pattern", [])
                        fill_legato = bool(fill_def.get("legato"))
                        fill_applied_this_iter = True
                        fade_beats_local = safe_get(
                            style_options,
                            "fade_beats",
                            default=self.fade_beats_default,
                            cast_to=float,
                            log_name=f"{log_render_prefix}.FadeBeats",
                        )
                        self.fill_offsets.append(
                            (
                                offset_in_score + current_pos_within_block,
                                fade_beats_local,
                            )
                        )
                fill_interval_bars = safe_get(
                    drums_params,
                    "drum_fill_interval_bars",
                    default=0,
                    cast_to=int,
                    log_name=f"{log_render_prefix}.FillInterval",
                )
                if (
                    not fill_applied_this_iter
                    and is_last_pattern_iteration_in_block
                    and fill_interval_bars > 0
                ):
                    if (ms_since_fill + 1) >= fill_interval_bars:
                        fill_keys_from_params = drums_params.get("drum_fill_keys", [])
                        possible_fills_for_style = [
                            fk for fk in fill_keys_from_params if fk in fills
                        ]
                        if possible_fills_for_style:
                            chosen_fill_key = self.rng.choice(possible_fills_for_style)
                            fill_def = self._get_effective_pattern_def(chosen_fill_key)
                            chosen_fill_pattern_list = fill_def.get(
                                chosen_fill_key, fill_def.get("pattern")
                            )
                            if chosen_fill_pattern_list is not None:
                                pattern_to_use_for_iteration = chosen_fill_pattern_list
                                fill_legato = bool(fill_def.get("legato"))
                                fill_applied_this_iter = True
                            logger.debug(
                                f"{log_render_prefix}: Applied scheduled fill '{chosen_fill_key}' for style '{style_key}'."
                            )
                start_bin = int(
                    (offset_in_score + current_pos_within_block)
                    * self.heatmap_resolution
                )
                end_bin = int(
                    (
                        offset_in_score
                        + current_pos_within_block
                        + current_pattern_iteration_ql
                    )
                    * self.heatmap_resolution
                )
                max_bin_val = 0
                for b in range(start_bin, end_bin):
                    max_bin_val = max(
                        max_bin_val, self.heatmap.get(b % self.heatmap_resolution, 0)
                    )
                velocity_scale = 1.2 if max_bin_val > self.heatmap_threshold else 1.0
                self._apply_pattern(
                    part,
                    pattern_to_use_for_iteration,
                    offset_in_score + current_pos_within_block,
                    current_pattern_iteration_ql,
                    base_vel,
                    swing_type,
                    swing_ratio_val,
                    pat_ts if pat_ts else self.global_ts,
                    drums_params,
                    velocity_scale,
                    velocity_curve_list,
                    legato=fill_legato,
                )
                if fill_applied_this_iter:
                    ms_since_fill = 0
                else:
                    ms_since_fill += 1
                current_pos_within_block += current_pattern_iteration_ql
                remaining_ql_in_block -= current_pattern_iteration_ql
                bars_since_section_start += 1

            running_beats += blk_data.get(
                "humanized_duration_beats",
                blk_data.get("q_length", pattern_unit_length_ql),
            )

        for off, fade_beats in self.fill_offsets:
            self._velocity_fade_into_fill(part, off, fade_beats)

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
    ) -> None:
        """Insert a list of drum events into ``part``.

        Each event may specify a ``type`` to trigger articulations such as
        ``drag`` (two grace notes), ``ruff`` (three grace notes), ``flam``
        (single grace) or ``ghost`` (low-velocity).

        Parameters
        ----------
        velocity_scale : float
            Multiplicative factor applied to every computed velocity.
        velocity_curve : list[float] | None
            Optional per-layer multipliers applied after ``velocity_scale``.
            When calling this method directly, pass ``velocity_scale`` first
            and then ``velocity_curve`` to avoid mis-scaled velocities.
        """
        log_apply_prefix = f"DrumGen.ApplyPattern"
        self.current_bpm = self._current_bpm(bar_start_abs_offset)
        if self.use_velocity_ema:
            self._vel_smoother.reset()
        beat_len_ql = (
            current_pattern_ts.beatDuration.quarterLength if current_pattern_ts else 1.0
        )
        if swing_type == "eighth":
            subdivision_duration_ql = beat_len_ql / 2.0
        elif swing_type == "sixteenth":
            subdivision_duration_ql = beat_len_ql / 4.0
        else:
            subdivision_duration_ql = beat_len_ql
        velocity_curve = velocity_curve or [1.0]

        vel_walk_state: int = drum_block_params.setdefault(
            "_vel_walk", pattern_base_velocity
        )

        if not events and self.groove_model:
            events = groove_sampler.generate_bar(
                prev_history=self._groove_history,
                model=self.groove_model,
                rng=self.rng,
                resolution=self.groove_resolution,
            )

        if self.consonant_peaks:
            start_sec = bar_start_abs_offset * 60.0 / self.current_bpm
            end_sec = (bar_start_abs_offset + current_bar_actual_len_ql) * 60.0 / self.current_bpm
            peaks_in_bar = [p - start_sec for p in self.consonant_peaks if start_sec <= p < end_sec]
            if peaks_in_bar:
                events = PeakSynchroniser.sync_events(
                    peaks_in_bar,
                    events,
                    tempo_bpm=self.current_bpm,
                    lag_ms=self.consonant_sync_cfg["lag_ms"],
                    min_distance_beats=self.consonant_sync_cfg["min_distance_beats"],
                    sustain_threshold_ms=self.consonant_sync_cfg["sustain_threshold_ms"],
                )

        prev_note: Optional[note.Note] = None
        prev_vel_scale = 1.0
        alpha = 0.5
        for ev_idx, ev_def in enumerate(events):
            log_event_prefix = f"{log_apply_prefix}.Evt{ev_idx}"
            if self.rng.random() > safe_get(
                ev_def,
                "probability",
                default=1.0,
                cast_to=float,
                log_name=f"{log_event_prefix}.Prob",
            ):
                continue
            inst_name = ev_def.get("instrument")
            if not inst_name:
                continue
            inst_name = MISSING_DRUM_MAP_FALLBACK.get(
                inst_name.lower(), inst_name.lower()
            )
            inst_name = DRUM_ALIAS.get(inst_name, inst_name).lower()
            articulation = str(ev_def.get("articulation", "")).lower()
            if articulation == "bell":
                inst_name = "ride_bell"
            elif articulation == "splash":
                inst_name = "splash"
            elif articulation == "choke":
                inst_name = "crash_choke"
            brush_active = self.drum_brush or (
                self.overrides and getattr(self.overrides, "drum_brush", False)
            )
            if inst_name not in self.gm_pitch_map:
                if self.strict_drum_map:
                    raise KeyError(f"Unknown drum instrument: '{inst_name}'")
                if inst_name not in self._warned_missing_drum_map:
                    logger.warning(f"Unknown drum instrument: '{inst_name}'")
                    self._warned_missing_drum_map.add(inst_name)

            rel_offset_in_pattern = safe_get(
                ev_def,
                "offset",
                default=0.0,
                cast_to=float,
                log_name=f"{log_event_prefix}.Offset",
            )
            event_bpm = self._current_bpm(bar_start_abs_offset + rel_offset_in_pattern)
            self.current_bpm = event_bpm
            blend = _combine_timing(
                rel_offset_in_pattern,
                beat_len_ql,
                swing_ratio=swing_ratio,
                swing_type=swing_type,
                push_pull_curve=self.current_push_pull_curve,
                tempo_bpm=event_bpm,
                max_push_ms=self.push_pull_max_ms,
                vel_range=self.velocity_range,
                return_vel=True,
            )
            rel_offset_in_pattern = blend.offset_ql
            vel_mul = prev_vel_scale * (1 - alpha) + blend.vel_scale * alpha
            prev_vel_scale = blend.vel_scale
            if rel_offset_in_pattern >= current_bar_actual_len_ql - (
                MIN_NOTE_DURATION_QL / 16.0
            ):
                continue

            velocity_idx = int(rel_offset_in_pattern / subdivision_duration_ql)

            hit_duration_ql_from_def = safe_get(
                ev_def,
                "duration",
                default=0.125,
                cast_to=float,
                log_name=f"{log_event_prefix}.Dur",
            )
            clipped_duration_ql = min(
                hit_duration_ql_from_def,
                current_bar_actual_len_ql - rel_offset_in_pattern,
            )
            if clipped_duration_ql < MIN_NOTE_DURATION_QL / 8.0:
                continue

            if self.random_walk_step:
                step = self.rng.randint(-4, 4)
                vel_walk_state = max(1, min(127, vel_walk_state + step))
                drum_block_params["_vel_walk"] = vel_walk_state
                base_vel_for_hit = vel_walk_state
            else:
                base_vel_for_hit = pattern_base_velocity

            if ev_def.get("type") == "ghost":
                final_event_velocity = int(pattern_base_velocity * 0.2)
                clipped_duration_ql = min(clipped_duration_ql, 0.1)
            else:
                final_event_velocity = safe_get(
                    ev_def,
                    "velocity",
                    default=int(
                        base_vel_for_hit
                        * safe_get(
                            ev_def,
                            "velocity_factor",
                            default=1.0,
                            cast_to=float,
                            log_name=f"{log_event_prefix}.VelFactor",
                        )
                    ),
                    cast_to=int,
                    log_name=f"{log_event_prefix}.VelAbs",
                )

            if articulation == "bell":
                final_event_velocity = min(127, int(final_event_velocity * 1.15))
            if articulation == "splash":
                max_dur = (0.3 * self.current_bpm) / 60.0
                clipped_duration_ql = min(clipped_duration_ql, max_dur)
            final_insert_offset_in_score = bar_start_abs_offset + rel_offset_in_pattern
            bin_idx = (
                int((final_insert_offset_in_score * self.heatmap_resolution))
                % self.heatmap_resolution
            )
            bin_count = self.heatmap.get(bin_idx, 0)
            rel = bin_count / self.max_heatmap_value if self.max_heatmap_value else 0

            if (
                inst_name in {"ghost", "ghost_hat"}
                and bin_count >= self.heatmap_threshold
            ):
                logger.debug(
                    f"{log_event_prefix}: Skip ghost hat at {final_insert_offset_in_score:.3f} (bin {bin_idx} count {bin_count})"
                )
                continue
            if inst_name in {"ghost", "ghost_hat"}:
                density = self.accent_mapper.ghost_density(rel)
                if not self.ghost_hat_on_offbeat:
                    beat_pos = (final_insert_offset_in_score * 2) % 1.0
                    if abs(beat_pos) < 1e-3:
                        continue
                if self.rng.random() > density:
                    continue

            if inst_name in {"kick", "snare"}:
                final_event_velocity = self.accent_mapper.accent(
                    rel, final_event_velocity
                )

            layer_idx = ev_def.get("velocity_layer")
            if velocity_curve and layer_idx is not None:
                try:
                    idx = int(layer_idx)
                    if 0 <= idx < len(velocity_curve):
                        final_event_velocity = int(
                            final_event_velocity * velocity_curve[idx]
                        )
                except (TypeError, ValueError):
                    pass

            final_event_velocity = max(
                1,
                min(
                    127,
                    int(final_event_velocity * velocity_scale * vel_mul),
                ),
            )
            if self.use_velocity_ema:
                final_event_velocity = self._vel_smoother.update(final_event_velocity)

            if brush_active and inst_name == "snare":
                inst_name = "snare_brush"
                final_event_velocity = max(1, int(final_event_velocity * 0.6))

            ev_type = ev_def.get("type")
            if ev_type in {"drag", "ruff"}:
                midi_pitch = self.gm_pitch_map.get(inst_name)
                if midi_pitch is not None:
                    self._insert_grace_chain(
                        part,
                        final_insert_offset_in_score,
                        midi_pitch,
                        final_event_velocity,
                        2 if ev_type == "drag" else 3,
                        spread_ms=float(ev_def.get("spread_ms", 25.0)),
                        velocity_curve=ev_def.get("grace_curve"),
                        humanize=ev_def.get("humanize_grace"),
                        tempo_bpm=event_bpm,
                    )
                continue

            if ev_def.get("type") == "flam":
                midi_pitch = self.gm_pitch_map.get(inst_name)
                if midi_pitch is not None:
                    self._insert_flam(
                        part,
                        final_insert_offset_in_score,
                        midi_pitch,
                        final_event_velocity,
                    )
                continue

            drum_hit_note = self._make_hit(
                inst_name, final_event_velocity, clipped_duration_ql, ev_def
            )
            if not drum_hit_note:
                continue

            if inst_name in {"ghost_snare", "ghost_tom"}:
                drum_hit_note = humanizer.apply_ghost_jitter(
                    drum_hit_note, self.rng, tempo_bpm=event_bpm
                )

            if ev_def.get("humanize_template") == "flam_legato_ghost":
                drum_hit_note = apply_humanization_to_element(
                    drum_hit_note, "flam_legato_ghost"
                )

            # New multi-template humanization
            templates = ev_def.get("humanize_templates")
            if templates:
                mode = str(ev_def.get("humanize_templates_mode", "sequential")).lower()
                if isinstance(templates, Sequence) and not isinstance(
                    templates, (str, bytes)
                ):
                    template_list = list(templates)
                else:
                    template_list = [templates]
                if mode == "random":
                    chosen = self.rng.choice(template_list)
                    drum_hit_note = apply_humanization_to_element(drum_hit_note, chosen)
                else:
                    for t_name in template_list:
                        drum_hit_note = apply_humanization_to_element(
                            drum_hit_note, t_name
                        )

            # (ヒューマナイズ処理は前回と同様)
            humanize_this_hit = False
            humanize_template_for_hit = "drum_tight"
            humanize_custom_for_hit = {}
            event_humanize_setting = ev_def.get("humanize")
            if isinstance(event_humanize_setting, bool):
                humanize_this_hit = event_humanize_setting
            elif isinstance(event_humanize_setting, str):
                humanize_this_hit = True
                humanize_template_for_hit = event_humanize_setting
            elif isinstance(event_humanize_setting, dict):
                humanize_this_hit = True
                humanize_template_for_hit = event_humanize_setting.get(
                    "template_name", humanize_template_for_hit
                )
                humanize_custom_for_hit = event_humanize_setting.get(
                    "custom_params", {}
                )
            else:
                if drum_block_params.get("humanize_opt", False):
                    humanize_this_hit = True
                    humanize_template_for_hit = drum_block_params.get(
                        "template_name", "drum_tight"
                    )
                    humanize_custom_for_hit = drum_block_params.get("custom_params", {})
            time_delta_from_humanizer = 0.0
            if humanize_this_hit:
                drum_hit_note = apply_humanization_to_element(
                    drum_hit_note,
                    template_name=humanize_template_for_hit,
                    custom_params=humanize_custom_for_hit,
                )
            final_insert_offset_in_score += time_delta_from_humanizer
            drum_hit_note.offset = 0.0
            if inst_name in {"kick", "bd", "acoustic_bass_drum"}:
                self.kick_offsets.append(bar_start_abs_offset + rel_offset_in_pattern)
            if legato and prev_note is not None:
                prev_note.tie = HoldTie()
            part.insert(final_insert_offset_in_score, drum_hit_note)

            if inst_name == "ohh" and self.rng.random() < self.open_hat_choke_prob:
                min_b = (self.open_hat_choke_min_ms / 1000.0) * (self.current_bpm / 60.0)
                max_b = (self.open_hat_choke_max_ms / 1000.0) * (self.current_bpm / 60.0)
                off = final_insert_offset_in_score + self.rng.uniform(min_b, max_b)
                vel = max(1, int(drum_hit_note.volume.velocity * self.rng.uniform(0.6, 0.9)))
                pedal_pitch = self.gm_pitch_map.get("hh_pedal")
                if pedal_pitch is not None:
                    q_off = PeakSynchroniser._quantize(off)
                    duplicate = False
                    for n in part.flatten().notes:
                        if (
                            PeakSynchroniser._quantize(float(n.offset)) == q_off
                            and n.pitch.midi == pedal_pitch
                        ):
                            duplicate = True
                            break
                    if not duplicate:
                        pedal_note = self._make_hit("hh_pedal", vel, MIN_NOTE_DURATION_QL / 8.0, None)
                        if pedal_note:
                            pedal_note.offset = 0.0
                            part.insert(off, pedal_note)

            if articulation == "choke":
                off = final_insert_offset_in_score + (0.2 * self.current_bpm / 60.0)
                note_off = note.Note()
                note_off.pitch = drum_hit_note.pitch
                note_off.duration = m21duration.Duration(0)
                note_off.volume = m21volume.Volume(velocity=0)
                note_off.offset = 0.0
                part.insert(off, note_off)
            prev_note = drum_hit_note
            mapped_name_for_history = getattr(
                drum_hit_note.editorial, "mapped_name", inst_name
            )
            self._groove_history.append(mapped_name_for_history)

        n_hist = int(self.groove_model.get("n", 3)) if self.groove_model else 0
        if n_hist > 1 and len(self._groove_history) > n_hist - 1:
            self._groove_history = self._groove_history[-(n_hist - 1) :]

    def _make_hit(
        self, name: str, vel: int, ql: float, ev_def: Optional[Dict[str, Any]] = None
    ) -> Optional[note.Note]:
        """Return a ``music21.note.Note`` for a single drum hit.

        Parameters
        ----------
        name : str
            Drum instrument label.
        vel : int
            MIDI velocity (1-127).
        ql : float
            Duration in quarterLength units.
        ev_def : dict | None
            Original event definition to inspect flags such as ``pedal``.

        Returns
        -------
        Optional[note.Note]
            Prepared note object or ``None`` if the sound is unknown. The
            resulting note stores its mapped label in ``note.editorial.mapped_name``.
        """
        mapped_name = name.lower().replace(" ", "_").replace("-", "_")
        brush_active = self.drum_brush or (
            self.overrides and getattr(self.overrides, "drum_brush", False)
        )
        if brush_active and mapped_name in BRUSH_MAP:
            mapped_name = BRUSH_MAP[mapped_name]
            vel = max(1, int(vel * 0.6))
        if mapped_name in {"chh", "hh", "hat_closed"} and vel < self.hh_edge_threshold:
            mapped_name = "hh_edge"
        if ev_def and ev_def.get("pedal"):
            mapped_name = "hh_pedal"
        actual_name_for_midi = GHOST_ALIAS.get(mapped_name, mapped_name)
        midi_pitch_val = self.gm_pitch_map.get(actual_name_for_midi)
        if midi_pitch_val is None:
            logger.warning(
                f"DrumGen _make_hit: Unknown drum sound '{name}' (mapped to '{actual_name_for_midi}'). MIDI mapping not found. Skipping."
            )
            return None
        n = note.Note()
        n.pitch = pitch.Pitch(midi=midi_pitch_val)
        n.duration = m21duration.Duration(
            quarterLength=max(MIN_NOTE_DURATION_QL / 8.0, ql)
        )
        n.volume = m21volume.Volume(velocity=max(1, min(127, vel)))
        n.offset = 0.0
        n.editorial.mapped_name = mapped_name
        return n

    def _insert_flam(
        self, part: stream.Part, offset: float, midi_pitch: int, velocity: int
    ) -> None:
        """Insert a flam consisting of a grace note before the main hit."""
        grace_offset = (30.0 / 1000.0) * (self.current_bpm / 60.0)
        grace = note.Note()
        grace.pitch = pitch.Pitch(midi=midi_pitch)
        grace.duration = m21duration.Duration(max(MIN_NOTE_DURATION_QL / 8.0, 0.05))
        grace.volume = m21volume.Volume(velocity=max(1, int(velocity * 0.4)))
        grace.offset = 0.0
        part.insert(max(0.0, offset - grace_offset), grace)
        main = note.Note()
        main.pitch = pitch.Pitch(midi=midi_pitch)
        main.duration = m21duration.Duration(max(MIN_NOTE_DURATION_QL / 8.0, 0.1))
        main.volume = m21volume.Volume(velocity=max(1, velocity))
        main.offset = 0.0
        part.insert(offset, main)

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
    ) -> None:
        """Insert a drag or ruff before the main hit.

        Parameters
        ----------
        part : :class:`music21.stream.Part`
            Part to insert the notes into.
        offset : float
            Offset of the main hit in quarterLength.
        midi_pitch : int
            MIDI pitch for all notes.
        velocity : int
            Velocity of the main hit.
        n_hits : int, optional
            Number of grace notes preceding the main hit.
        spread_ms : float, optional
            Duration of the grace window in milliseconds.

        Notes
        -----
        Grace notes are spaced evenly within ``spread_ms`` before ``offset`` and
        their velocities fade from roughly 30%% to 60%% of ``velocity``. ``velocity_curve``
        may be ``"exp"`` for exponential fade or a list of multipliers.
        """
        window_ms = max(MIN_GRACE_MS, min(MAX_GRACE_MS, float(spread_ms)))
        step_ms = window_ms / max(1, n_hits)

        def _get_factor(i: int) -> float:
            if isinstance(velocity_curve, Sequence):
                if i < len(velocity_curve):
                    return float(velocity_curve[i])
                return float(velocity_curve[-1]) if velocity_curve else 0.5
            if isinstance(velocity_curve, str) and velocity_curve.lower() == "exp":
                base = 0.3
                top = 0.6
                exponent = i / max(1, n_hits - 1)
                return base * ((top / base) ** exponent)
            return 0.3 + (0.3 * i / max(1, n_hits - 1))

        def _maybe_humanize(n: note.Note) -> note.Note:
            if not humanize:
                return n
            if isinstance(humanize, bool):
                tmpl = "drum_tight"
                return apply_humanization_to_element(n, template_name=tmpl)
            if isinstance(humanize, str):
                return apply_humanization_to_element(n, template_name=humanize)
            if isinstance(humanize, dict):
                tmpl = humanize.get("template_name")
                params = humanize.get("custom_params")
                return apply_humanization_to_element(
                    n, template_name=tmpl, custom_params=params
                )
            return n

        tempo = tempo_bpm or self.global_tempo
        for idx in range(n_hits):
            off_ms = window_ms - idx * step_ms
            grace_offset = (off_ms / 1000.0) * (tempo / 60.0)
            factor = _get_factor(idx)
            grace = note.Note()
            grace.pitch = pitch.Pitch(midi=midi_pitch)
            grace.duration = m21duration.Duration(max(MIN_NOTE_DURATION_QL / 8.0, 0.05))
            grace.volume = m21volume.Volume(velocity=max(1, int(velocity * factor)))
            grace.offset = 0.0
            grace = _maybe_humanize(grace)
            part.insert(max(0.0, offset - grace_offset), grace)
        main = note.Note()
        main.pitch = pitch.Pitch(midi=midi_pitch)
        main.duration = m21duration.Duration(max(MIN_NOTE_DURATION_QL / 8.0, 0.1))
        main.volume = m21volume.Volume(velocity=max(1, velocity))
        main.offset = 0.0
        part.insert(offset, main)

    def _load_vocal_end_times(self, midi_path: str) -> List[float]:
        if not midi_path or not Path(midi_path).exists():
            return []
        try:
            midi = converter.parse(midi_path)
            return sorted(
                float(n.offset + n.quarterLength)
                for n in midi.flatten().notes
                if isinstance(n, note.Note)
            )
        except Exception as exc:
            logger.warning(f"DrumGen: failed to load vocal MIDI for hi-hat sync: {exc}")
            return []

    def _velocity_fade_into_fill(
        self, part: stream.Part, fill_offset: float, fade_beats: float = 2.0
    ) -> None:
        """Gradually increase velocity leading into a fill.

        Parameters
        ----------
        part : stream.Part
            Part to apply the fade to.
        fill_offset : float
            Absolute offset where the fill begins.
        fade_beats : float, optional
            Length of the fade window in beats. Defaults to 2.0.
        """
        notes = [
            n
            for n in part.flatten().notes
            if fill_offset - fade_beats <= n.offset < fill_offset
        ]
        if len(notes) <= 1:
            return
        notes.sort(key=lambda n: n.offset)
        count = len(notes)
        for idx, n in enumerate(notes):
            base = (
                n.volume.velocity if n.volume and n.volume.velocity is not None else 64
            )
            scale = 0.8 + (0.2 * (idx + 1) / count)
            new_vel = int(max(1, min(127, base * scale)))
            if n.volume:
                n.volume.velocity = new_vel
            else:
                n.volume = m21volume.Volume(velocity=new_vel)

    def _apply_push_pull(self, rel_offset: float, beat_len_ql: float) -> float:
        curve = getattr(self, "current_push_pull_curve", None)
        if not curve:
            return rel_offset
        try:
            beat_idx = int(rel_offset / beat_len_ql)
            shift_ms = float(curve[beat_idx])
        except (ValueError, IndexError, TypeError):
            return rel_offset
        shift_ql = (shift_ms / 1000.0) * (self.current_bpm / 60.0)
        return max(0.0, rel_offset + shift_ql)

    def _sync_hihat_with_vocals(self, part: stream.Part) -> None:
        if not self.vocal_end_times:
            return
        chh = self.gm_pitch_map.get("chh")
        ohh = self.gm_pitch_map.get("ohh")
        if chh is None or ohh is None:
            return
        thr = (50.0 / 1000.0) * (self.current_bpm / 60.0)
        beat_len = self.global_ts.beatDuration.quarterLength
        bar_len = self.global_ts.barDuration.quarterLength
        notes = sorted(part.recurse().notes, key=lambda n: n.offset)
        idx = 0
        for end in self.vocal_end_times:
            beat4 = round(end / bar_len) * bar_len
            if abs(end - beat4) > thr:
                continue
            while idx < len(notes) and notes[idx].offset < end - 1e-6:
                idx += 1
            while idx < len(notes):
                n = notes[idx]
                if n.offset >= end - 1e-6:
                    if int(n.pitch.midi) == chh:
                        n.pitch.midi = ohh
                        idx += 1
                        break
                idx += 1

    def _add_internal_default_patterns(self):
        """ライブラリに必須パターンがなければ、最低限のフォールバックを追加"""
        defaults = {
            "default_drum_pattern": {
                "pattern": [
                    {"offset": 0.0, "instrument": "kick"},
                    {"offset": 2.0, "instrument": "snare"},
                ],
                "length_beats": 4.0,
            },
            "no_drums": {"pattern": [], "length_beats": 4.0},
            "no_drums_or_sparse_cymbal": {
                "pattern": [
                    {"offset": 0.0, "instrument": "crash", "velocity_factor": 0.5}
                ],
                "length_beats": 4.0,
            },
            "ballad_soft_kick_snare_8th_hat": {
                "pattern": [
                    {"offset": 0, "instrument": "kick"},
                    {"offset": 2, "instrument": "snare"},
                ],
                "length_beats": 4.0,
            },
            "rock_beat_A_8th_hat": {
                "pattern": [
                    {"offset": 0, "instrument": "kick"},
                    {"offset": 1, "instrument": "chh"},
                    {"offset": 2, "instrument": "snare"},
                    {"offset": 3, "instrument": "chh"},
                ],
                "length_beats": 4.0,
            },
            "rock_ballad_build_up_8th_hat": {
                "pattern": [{"offset": i * 0.5, "instrument": "chh"} for i in range(8)],
                "length_beats": 4.0,
            },
            "anthem_rock_chorus_16th_hat": {
                "pattern": [
                    {"offset": i * 0.25, "instrument": "chh"} for i in range(16)
                ],
                "length_beats": 4.0,
            },
            "anthem_rock_chorus_16th_hat_fill": {
                "pattern": [
                    {"offset": i * 0.25, "instrument": "snare"} for i in range(16)
                ],
                "length_beats": 4.0,
            },
        }
        for key, val in defaults.items():
            if key not in self.part_parameters:
                self.part_parameters[key] = val

    def _load_pattern_lib(self, paths: List[str | Path]) -> Dict[str, Any]:
        """Load drum pattern definitions from YAML files.

        Parameters
        ----------
        paths : list of str or Path
            One or more YAML files containing pattern definitions. Each file may
            contain multiple YAML documents. Documents may either provide a top
            level ``drum_patterns`` mapping or the mapping itself.

        Returns
        -------
        Dict[str, Any]
            Combined pattern dictionary keyed by style name.
        """

        library: Dict[str, Any] = {}
        for p in paths:
            p = Path(p)
            if not p.is_absolute():
                repo_root = Path(__file__).resolve().parents[1]
                p = repo_root / p
            try:
                with p.open("r", encoding="utf-8") as fh:
                    for doc in yaml.safe_load_all(fh):
                        if not isinstance(doc, dict):
                            continue
                        if "drum_patterns" in doc and isinstance(
                            doc["drum_patterns"], dict
                        ):
                            library.update(doc["drum_patterns"])
                        else:
                            library.update(doc)
            except FileNotFoundError:
                logger.warning(f"DrumGen _load_pattern_lib: file not found: {p}")
            except Exception as exc:
                logger.warning(
                    f"DrumGen _load_pattern_lib: failed to load '{p}': {exc}"
                )
        return library

    def _resolve_style_key(
        self,
        musical_intent: Dict[str, Any],
        overrides: Dict[str, Any],
        section_data: Optional[Dict[str, Any]] = None,
    ) -> str:
        """オーバーライドと感情から最終的なリズムキーを決定する"""
        if overrides and overrides.get("rhythm_key"):
            return overrides["rhythm_key"]

        expr = None
        if section_data:
            expr = section_data.get("expression_details")
        if not expr:
            expr = musical_intent.get("expression_details")
        if expr:
            key = (expr.get("emotion_bucket"), expr.get("intensity"))
            lut_style = EMOTION_INTENSITY_LUT.get(key)
            if lut_style and lut_style in self.part_parameters:
                return lut_style

        emotion = musical_intent.get("emotion", "default").lower()
        intensity = musical_intent.get("intensity", "medium").lower()

        return self._choose_pattern_key(emotion, intensity, musical_intent)

    def _render_part(
        self,
        section_data: Dict[str, Any],
        next_section_data: Optional[Dict[str, Any]] = None,
    ) -> stream.Part:
        """Generate a drum part for a single section."""
        part = stream.Part(id=self.part_name)
        part.insert(0, self.default_instrument)

        drum_params = section_data.setdefault("part_params", {}).setdefault(
            self.part_name, {}
        )
        musical_intent = section_data.get("musical_intent", {})
        style_key = self._resolve_style_key(
            musical_intent,
            drum_params,
            section_data,
        )
        drum_params.setdefault("final_style_key_for_render", style_key)

        self._render([section_data], part, section_data)
        return part

    def get_kick_offsets(self) -> List[float]:
        return list(self.kick_offsets)

    def get_fill_offsets(self) -> List[float]:
        return [
            off if not isinstance(off, tuple) else off[0] for off in self.fill_offsets
        ]


