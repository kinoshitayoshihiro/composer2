# --- START OF FILE generator/drum_generator.py (最終FIX版) ---
from __future__ import annotations
import logging, random, json, copy
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional
from utilities.velocity_curve import resolve_velocity_curve
from music21 import (
    stream,
    note,
    pitch,
    meter,
    instrument as m21instrument,
    duration as m21duration,
    volume as m21volume,
    tie,
)

from .base_part_generator import BasePartGenerator
from utilities.core_music_utils import get_time_signature_object, MIN_NOTE_DURATION_QL
from utilities.onset_heatmap import build_heatmap, RESOLUTION, load_heatmap
from utilities import humanizer
from utilities.humanizer import apply_humanization_to_element
from utilities.safe_get import safe_get
from utilities.drum_map_registry import (
    DRUM_MAP,
    get_drum_map,
    GM_DRUM_MAP,
    MISSING_DRUM_MAP_FALLBACK,
)


logger = logging.getLogger("modular_composer.drum_generator")

# Hat suppression: omit hi-hat hits when relative vocal activity exceeds this
# threshold (0-1 scale based on heatmap weight).
HAT_SUPPRESSION_THRESHOLD = 0.6

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
}
GHOST_ALIAS: Dict[str, str] = {"ghost_snare": "snare", "gs": "snare"}
BRUSH_MAP: Dict[str, str] = {"kick": "brush_kick", "snare": "brush_snare"}


class AccentMapper:
    """Map accent strength and ghost-hat density using vocal heatmap."""

    def __init__(self, threshold: float = 0.6, ghost_density_range=(0.3, 0.8)) -> None:
        self.threshold = threshold
        self.ghost_density_range = ghost_density_range

    def accent(self, rel: float, velocity: int) -> int:
        return self.get_velocity(rel, velocity)

    def get_velocity(self, rel: float, base_velocity: int) -> int:
        if rel >= self.threshold:
            return min(127, int(base_velocity * 1.2))
        return base_velocity

    def ghost_density(self, rel: float) -> float:
        low, high = self.ghost_density_range
        return high if rel < self.threshold else low


class FillInserter:
    """Insert drum fills at section boundaries."""

    def __init__(self, pattern_lib: Dict[str, Any]) -> None:
        self.pattern_lib = pattern_lib

    def insert(
        self, part: stream.Part, section_data: Dict[str, Any], fill_key: Optional[str] = None
    ) -> None:
        key = fill_key or section_data.get("drum_fill_at_end")
        if not key:
            return
        fill_def = self.pattern_lib.get(key)
        if fill_def is None:
            logger.warning("FillInserter.insert: fill pattern '%s' not found", key)
            return
        events = fill_def.get("pattern", [])
        if not events:
            return
        start = (
            section_data.get("absolute_offset", 0.0)
            + section_data.get("q_length", 4.0)
            - 4.0
        )
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
            n.duration = m21duration.Duration(ev.get("duration", 0.25))
            n.volume = m21volume.Volume(
                velocity=int(80 * ev.get("velocity_factor", 1.0))
            )
            part.insert(start + float(ev.get("offset", 0.0)), n)

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
        **kwargs,
    ):
        self.main_cfg = main_cfg
        super().__init__(
            global_settings=global_settings,
            default_instrument=default_instrument,
            global_tempo=global_tempo,
            global_time_signature=global_time_signature,
            global_key_signature_tonic=global_key_signature_tonic,
            global_key_signature_mode=global_key_signature_mode,
            **kwargs,
        )
        # ここに他の初期化処理をまとめて書く
        self.logger = logging.getLogger("modular_composer.drum_generator")
        self.part_parameters = kwargs.get("part_parameters", {})
        self.kick_offsets: List[float] = []
        self.fill_offsets: List[float] = []
        self.strict_drum_map = bool((global_settings or {}).get("strict_drum_map", False))
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

        # Use path settings from main_cfg, falling back to general paths
        self.vocal_midi_path = (
            self.main_cfg.get("vocal_midi_path_for_drums")
            or self.main_cfg.get("paths", {}).get("vocal_note_data_path")
        )

        heatmap_json_path = self.main_cfg.get("heatmap_json_path_for_drums")
        if not heatmap_json_path:
            heatmap_json_path = self.main_cfg.get("paths", {}).get(
                "vocal_heatmap_path"
            )
        if not heatmap_json_path:
            heatmap_json_path = str(Path("data/heatmap.json").resolve())

        heatmap_json_path = str(Path(heatmap_json_path).expanduser().resolve())
        self.heatmap = load_heatmap(heatmap_json_path)
        self.max_heatmap_value = max(self.heatmap.values()) if self.heatmap else 0
        self.heatmap_resolution = self.main_cfg.get("heatmap_resolution", RESOLUTION)
        self.heatmap_threshold = self.main_cfg.get("heatmap_threshold", 1)
        self.rng = random.Random()
        if self.main_cfg.get("rng_seed") is not None:
            self.rng.seed(self.main_cfg["rng_seed"])

        self.accent_mapper = AccentMapper(
            self.main_cfg.get("accent_threshold", 0.6),
            tuple(self.main_cfg.get("ghost_density_range", [0.3, 0.8])),
        )
        self.ghost_hat_on_offbeat = self.main_cfg.get("ghost_hat_on_offbeat", True)
        self.drum_brush = bool(self.main_cfg.get("drum_brush", False))

        # apply groove pretty
        global_cfg = self.main_cfg.get("global_settings", {})
        groove_path = global_cfg.get("groove_profile_path")
        self.groove_strength = float(global_cfg.get("groove_strength", 1.0))
        self.groove_profile = {}
        if groove_path:
            try:
                with open(groove_path, "r", encoding="utf-8") as f:
                    self.groove_profile = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load groove profile from {groove_path}: {e}")

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
        lib = yaml.safe_load(
            open(self.main_cfg["paths"]["rhythm_library_path"], "r", encoding="utf-8")
        )
        self.raw_pattern_lib.update(lib.get("drum_patterns", {}))

        # 最終的なパターン辞書を part_parameters に適用
        self.part_parameters = self.raw_pattern_lib

    def _choose_pattern_key(
        self,
        emotion: str | None,
        intensity: str | None,
        musical_intent: Optional[Dict[str, Any]] = None,
    ) -> str:
        emo = (emotion or "default").lower()
        inten = (intensity or "medium").lower()
        bucket = EMO_TO_BUCKET.get(emo, "groove")
        base_key = BUCKET_INT_TO_PATTERN.get((bucket, inten), "groove_pocket")
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
        self.heatmap_resolution = (
            (self.main_cfg or {}).get("heatmap_resolution")
            or self.global_settings.get("heatmap_resolution", RESOLUTION)
        )
        self.heatmap_threshold = (
            (self.main_cfg or {}).get("heatmap_threshold")
            or self.global_settings.get("heatmap_threshold", 0.5)
        )

        if section_data and section_data.get("expression_details"):
            expr = section_data["expression_details"]
            key = (expr.get("emotion_bucket"), expr.get("intensity"))
            mapped = EMOTION_INTENSITY_LUT.get(key)
            if mapped:
                section_data.setdefault("part_params", {}).setdefault(self.part_name, {})[
                    "rhythm_key"
                ] = mapped

        part = super().compose(
            section_data=section_data,
            overrides_root=overrides_root,
            groove_profile_path=groove_profile_path,
            next_section_data=next_section_data,
            part_specific_humanize_params=part_specific_humanize_params,
            shared_tracks=shared_tracks,
        )

        if section_data:
            self.fill_inserter.insert(part, section_data)
        return part

    def _render(
        self,
        blocks: Sequence[Dict[str, Any]],
        part: stream.Part,
    ):
        ms_since_fill = 0
        bars_since_section_start = 0
        for blk_idx, blk_data in enumerate(blocks):
            log_render_prefix = f"DrumGen.Render.Blk{blk_idx+1}"  # 1-indexed for logs
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
            velocity_curve_list = resolve_velocity_curve(style_options.get("velocity_curve"))

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

                preferred_positions = [int(p) for p in style_def.get("preferred_fill_positions", []) if isinstance(p, int)]
                fill_keys = style_def.get("fill_patterns", [])
                at_section_end = blk_idx == len(blocks) - 1 and is_last_pattern_iteration_in_block
                bar_number = bars_since_section_start + 1
                if (
                    not fill_applied_this_iter
                    and fill_keys
                    and (bar_number in preferred_positions or at_section_end)
                ):
                    candidates = [
                        fk
                        for fk in fill_keys
                        if self._get_effective_pattern_def(fk).get("length_beats", pattern_unit_length_ql)
                        == pattern_unit_length_ql
                    ]
                    if candidates:
                        fill_key = self.rng.choice(candidates)
                        fill_def = self._get_effective_pattern_def(fill_key)
                        pattern_to_use_for_iteration = fill_def.get("pattern", [])
                        fill_legato = bool(fill_def.get("legato"))
                        fill_applied_this_iter = True
                        self.fill_offsets.append(offset_in_score + current_pos_within_block)
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
                            chosen_fill_pattern_list = fill_def.get("pattern", [])
                            if chosen_fill_pattern_list is not None:
                                pattern_to_use_for_iteration = chosen_fill_pattern_list
                                fill_legato = bool(fill_def.get("legato"))
                                fill_applied_this_iter = True
                                logger.debug(
                                    f"{log_render_prefix}: Applied scheduled fill '{chosen_fill_key}' for style '{style_key}'."
                                )
                start_bin = int((offset_in_score + current_pos_within_block) * self.heatmap_resolution)
                end_bin = int((offset_in_score + current_pos_within_block + current_pattern_iteration_ql) * self.heatmap_resolution)
                max_bin_val = 0
                for b in range(start_bin, end_bin):
                    max_bin_val = max(max_bin_val, self.heatmap.get(b % self.heatmap_resolution, 0))
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
    ):
        log_apply_prefix = f"DrumGen.ApplyPattern"
        beat_len_ql = (
            current_pattern_ts.beatDuration.quarterLength if current_pattern_ts else 1.0
        )

        prev_note: Optional[note.Note] = None
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
            inst_name = MISSING_DRUM_MAP_FALLBACK.get(inst_name.lower(), inst_name.lower())
            inst_name = DRUM_ALIAS.get(inst_name, inst_name).lower()
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
            if abs(swing_ratio - 0.5) > 1e-3:
                rel_offset_in_pattern = self._swing(
                    rel_offset_in_pattern, swing_ratio, beat_len_ql, swing_type
                )
            if rel_offset_in_pattern >= current_bar_actual_len_ql - (
                MIN_NOTE_DURATION_QL / 16.0
            ):
                continue

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

            if ev_def.get("type") == "ghost":
                final_event_velocity = int(pattern_base_velocity * 0.2)
                clipped_duration_ql = min(clipped_duration_ql, 0.1)
            else:
                final_event_velocity = safe_get(
                ev_def,
                "velocity",
                default=int(
                    pattern_base_velocity
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
            final_event_velocity = max(1, min(127, final_event_velocity))

            final_insert_offset_in_score = bar_start_abs_offset + rel_offset_in_pattern
            bin_idx = int((final_insert_offset_in_score * self.heatmap_resolution)) % self.heatmap_resolution
            bin_count = self.heatmap.get(bin_idx, 0)
            rel = bin_count / self.max_heatmap_value if self.max_heatmap_value else 0

            if inst_name in {"ghost", "ghost_hat"} and bin_count >= self.heatmap_threshold:
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
                final_event_velocity = self.accent_mapper.accent(rel, final_event_velocity)

            layer_idx = ev_def.get("velocity_layer")
            if velocity_curve and layer_idx is not None:
                try:
                    idx = int(layer_idx)
                    if 0 <= idx < len(velocity_curve):
                        final_event_velocity = int(final_event_velocity * velocity_curve[idx])
                except (TypeError, ValueError):
                    pass

            final_event_velocity = max(1, min(127, int(final_event_velocity * velocity_scale)))

            if ev_def.get("type") == "flam":
                midi_pitch = self.gm_pitch_map.get(inst_name)
                if midi_pitch is not None:
                    self._insert_flam(
                        part, final_insert_offset_in_score, midi_pitch, final_event_velocity
                    )
                continue

            drum_hit_note = self._make_hit(
                inst_name, final_event_velocity, clipped_duration_ql
            )
            if not drum_hit_note:
                continue

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
            if legato and prev_note is not None:
                prev_note.tie = tie.Tie('hold')
            part.insert(final_insert_offset_in_score, drum_hit_note)
            prev_note = drum_hit_note

    def _swing(
        self,
        rel_offset: float,
        swing_ratio: float,
        beat_len_ql: float,
        swing_type: str = "eighth",
    ) -> float:
        # (前回と同様)
        if abs(swing_ratio - 0.5) < 1e-3 or beat_len_ql <= 0:
            return rel_offset
        subdivision_duration_ql: float
        if swing_type == "eighth":
            subdivision_duration_ql = beat_len_ql / 2.0
        elif swing_type == "sixteenth":
            subdivision_duration_ql = beat_len_ql / 4.0
        else:
            logger.warning(
                f"DrumGen _swing: Unsupported swing_type '{swing_type}'. No swing."
            )
            return rel_offset
        if subdivision_duration_ql <= 0:
            return rel_offset
        effective_beat_for_swing_pair_ql = subdivision_duration_ql * 2.0
        beat_num_in_bar_for_swing_pair = math.floor(
            rel_offset / effective_beat_for_swing_pair_ql
        )
        offset_within_effective_beat = rel_offset - (
            beat_num_in_bar_for_swing_pair * effective_beat_for_swing_pair_ql)
        epsilon = subdivision_duration_ql * 0.1
        if abs(offset_within_effective_beat - subdivision_duration_ql) < epsilon:
            new_offset_within_effective_beat = (
                effective_beat_for_swing_pair_ql * swing_ratio
            )
            swung_rel_offset = (
                beat_num_in_bar_for_swing_pair * effective_beat_for_swing_pair_ql
            ) + new_offset_within_effective_beat
            return swung_rel_offset
        return rel_offset

    def _make_hit(self, name: str, vel: int, ql: float) -> Optional[note.Note]:
        # (前回と同様)
        mapped_name = name.lower().replace(" ", "_").replace("-", "_")
        if self.drum_brush and mapped_name in BRUSH_MAP:
            mapped_name = BRUSH_MAP[mapped_name]
        actual_name_for_midi = GHOST_ALIAS.get(mapped_name, mapped_name)
        midi_pitch_val = self.gm_pitch_map.get(actual_name_for_midi)
        if midi_pitch_val is None:
            logger.warning(
                f"DrumGen _make_hit: Unknown drum sound '{name}' (mapped to '{actual_name_for_midi}'). MIDI mapping not found. Skipping."
            )
            return None
        n = note.Note()
        n.pitch = pitch.Pitch(midi=midi_pitch_val)
        n.duration = m21duration.Duration(quarterLength=max(MIN_NOTE_DURATION_QL / 8.0, ql))
        n.volume = m21volume.Volume(velocity=max(1, min(127, vel)))
        n.offset = 0.0
        return n

    def _insert_flam(
        self, part: stream.Part, offset: float, midi_pitch: int, velocity: int
    ) -> None:
        """Insert a flam consisting of a grace note before the main hit."""
        grace_offset = (30.0 / 1000.0) * (self.global_tempo / 60.0)
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
                        if "drum_patterns" in doc and isinstance(doc["drum_patterns"], dict):
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

        emotion = musical_intent.get("emotion", "default")
        intensity = musical_intent.get("intensity", "medium")
        return self._choose_pattern_key(emotion, intensity, musical_intent)

    def _render_part(
        self,
        section_data: Dict[str, Any],
        next_section_data: Optional[Dict[str, Any]] = None,
    ) -> stream.Part:
        """単一のコードブロックに対してドラムパートを生成する"""
        part = stream.Part(id=self.part_name)
        part.insert(0, self.default_instrument)

        drum_params = section_data.get("part_params", {}).get(self.part_name, {})
        musical_intent = section_data.get("musical_intent", {})

        style_key = self._resolve_style_key(
            musical_intent,
            drum_params,
            section_data,
        )
        pattern_def = self.part_parameters.get(style_key)

        if not pattern_def or not pattern_def.get("pattern"):
            logger.warning(
                f"Drum pattern for key '{style_key}' is empty or not found. Skipping block."
            )
            part.append(note.Rest(quarterLength=section_data.get("q_length", 4.0)))
            return part

        block_duration = section_data.get("q_length", 4.0)
        pattern_events = pattern_def.get("pattern", [])
        # apply groove pretty
        adjusted_events = []
        resolution = RESOLUTION
        for ev in pattern_events:
            ev_copy = ev.copy()
            bin_idx = int(float(ev_copy.get("offset", 0.0)) * resolution)
            groove_offset = self.groove_profile.get(str(bin_idx), 0)
            ev_copy["offset"] = float(ev_copy.get("offset", 0.0)) + groove_offset * self.groove_strength
            adjusted_events.append(ev_copy)
        pattern_events = adjusted_events
        pattern_ref_duration = float(pattern_def.get("length_beats", 4.0))
        time_scale_factor = (
            block_duration / pattern_ref_duration if pattern_ref_duration > 0 else 1.0
        )
        base_velocity = drum_params.get("velocity", 80)

        for event in pattern_events:
            event_offset = float(event.get("offset", 0.0))
            event_dur = float(event.get("duration", 0.1))
            inst_name = event.get("instrument")
            vel_factor = float(event.get("velocity_factor", 1.0))

            if not inst_name:
                continue
            gm_name, midi_pitch = self.drum_map.get(inst_name, (None, None))
            if midi_pitch is None:
                logger.warning("Unknown drum label %s", inst_name)
                continue
            inst_name = gm_name
            inst_name = MISSING_DRUM_MAP_FALLBACK.get(inst_name.lower(), inst_name.lower())
            inst_name = DRUM_ALIAS.get(inst_name, inst_name).lower()
            
            final_offset = event_offset * time_scale_factor
            if final_offset >= block_duration:
                continue

            final_dur = event_dur * time_scale_factor
            final_velocity = max(1, min(127, int(base_velocity * vel_factor)))

            bin_idx = int((final_offset * self.heatmap_resolution)) % self.heatmap_resolution
            heat_val = self.heatmap.get(bin_idx, 0)
            rel = heat_val / self.max_heatmap_value if self.max_heatmap_value else 0
            final_velocity = self.accent_mapper.get_velocity(rel, final_velocity)
            if "chh" in inst_name.lower():
                if self.rng.random() > self.accent_mapper.ghost_density(rel):
                    continue

            midi_pitch = self.gm_pitch_map.get(inst_name.lower())

            if midi_pitch:
                if inst_name.lower() in {"kick", "bd", "acoustic_bass_drum"}:
                    abs_off = section_data.get("absolute_offset", 0.0) + final_offset
                    self.kick_offsets.append(abs_off)
                n = note.Note()
                n.pitch = pitch.Pitch(midi=midi_pitch)
                n.duration.quarterLength = final_dur
                n.volume = m21volume.Volume(velocity=final_velocity)
                part.insert(final_offset, n)
            else:
                if self.strict_drum_map:
                    raise KeyError(f"Unknown drum instrument: '{inst_name}'")
                if inst_name not in self._warned_missing_drum_map:
                    logger.warning(f"Unknown drum instrument: '{inst_name}'")
                    self._warned_missing_drum_map.add(inst_name)

        profile_name = (
            (self.main_cfg or {}).get("humanize_profile")
            or section_data.get("humanize_profile")
            or self.global_settings.get("humanize_profile")
        )
        if profile_name:
            humanizer.apply(part, profile_name)

        return part

    def get_kick_offsets(self) -> List[float]:
        return list(self.kick_offsets)

    def get_fill_offsets(self) -> List[float]:
        return list(self.fill_offsets)


# --- END OF FILE generator/drum_generator.py ---

# --- START OF FILE generator/drum_generator.py (最終FIX版) ---
from __future__ import annotations
import logging, random, json, copy
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional
from utilities.velocity_curve import resolve_velocity_curve
from music21 import (
    stream,
    note,
    pitch,
    meter,
    instrument as m21instrument,
    duration as m21duration,
    volume as m21volume,
    tie,
)

from .base_part_generator import BasePartGenerator
from utilities.core_music_utils import get_time_signature_object, MIN_NOTE_DURATION_QL
from utilities.onset_heatmap import build_heatmap, RESOLUTION, load_heatmap
from utilities import humanizer
from utilities.humanizer import apply_humanization_to_element
from utilities.safe_get import safe_get
from utilities.drum_map_registry import (
    DRUM_MAP,
    get_drum_map,
    GM_DRUM_MAP,
    MISSING_DRUM_MAP_FALLBACK,
)


logger = logging.getLogger("modular_composer.drum_generator")

# Hat suppression: omit hi-hat hits when relative vocal activity exceeds this
# threshold (0-1 scale based on heatmap weight).
HAT_SUPPRESSION_THRESHOLD = 0.6

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
}
GHOST_ALIAS: Dict[str, str] = {"ghost_snare": "snare", "gs": "snare"}
BRUSH_MAP: Dict[str, str] = {"kick": "brush_kick", "snare": "brush_snare"}


class AccentMapper:
    """Map accent strength and ghost-hat density using vocal heatmap."""

    def __init__(self, threshold: float = 0.6, ghost_density_range=(0.3, 0.8)) -> None:
        self.threshold = threshold
        self.ghost_density_range = ghost_density_range

    def accent(self, rel: float, velocity: int) -> int:
        return self.get_velocity(rel, velocity)

    def get_velocity(self, rel: float, base_velocity: int) -> int:
        if rel >= self.threshold:
            return min(127, int(base_velocity * 1.2))
        return base_velocity

    def ghost_density(self, rel: float) -> float:
        low, high = self.ghost_density_range
        return high if rel < self.threshold else low


class FillInserter:
    """Insert drum fills at section boundaries."""

    def __init__(self, pattern_lib: Dict[str, Any]) -> None:
        self.pattern_lib = pattern_lib

    def insert(
        self, part: stream.Part, section_data: Dict[str, Any], fill_key: Optional[str] = None
    ) -> None:
        key = fill_key or section_data.get("drum_fill_at_end")
        if not key:
            return
        fill_def = self.pattern_lib.get(key)
        if fill_def is None:
            logger.warning("FillInserter.insert: fill pattern '%s' not found", key)
            return
        events = fill_def.get("pattern", [])
        if not events:
            return
        start = (
            section_data.get("absolute_offset", 0.0)
            + section_data.get("q_length", 4.0)
            - 4.0
        )
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
            n.duration = m21duration.Duration(ev.get("duration", 0.25))
            n.volume = m21volume.Volume(
                velocity=int(80 * ev.get("velocity_factor", 1.0))
            )
            part.insert(start + float(ev.get("offset", 0.0)), n)

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
        **kwargs,
    ):
        self.main_cfg = main_cfg
        super().__init__(
            global_settings=global_settings,
            default_instrument=default_instrument,
            global_tempo=global_tempo,
            global_time_signature=global_time_signature,
            global_key_signature_tonic=global_key_signature_tonic,
            global_key_signature_mode=global_key_signature_mode,
            **kwargs,
        )
        # ここに他の初期化処理をまとめて書く
        self.logger = logging.getLogger("modular_composer.drum_generator")
        self.part_parameters = kwargs.get("part_parameters", {})
        self.kick_offsets: List[float] = []
        self.fill_offsets: List[float] = []
        self.strict_drum_map = bool((global_settings or {}).get("strict_drum_map", False))
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

        # Use path settings from main_cfg, falling back to general paths
        self.vocal_midi_path = (
            self.main_cfg.get("vocal_midi_path_for_drums")
            or self.main_cfg.get("paths", {}).get("vocal_note_data_path")
        )

        heatmap_json_path = self.main_cfg.get("heatmap_json_path_for_drums")
        if not heatmap_json_path:
            heatmap_json_path = self.main_cfg.get("paths", {}).get(
                "vocal_heatmap_path"
            )
        if not heatmap_json_path:
            heatmap_json_path = str(Path("data/heatmap.json").resolve())

        heatmap_json_path = str(Path(heatmap_json_path).expanduser().resolve())
        self.heatmap = load_heatmap(heatmap_json_path)
        self.max_heatmap_value = max(self.heatmap.values()) if self.heatmap else 0
        self.heatmap_resolution = self.main_cfg.get("heatmap_resolution", RESOLUTION)
        self.heatmap_threshold = self.main_cfg.get("heatmap_threshold", 1)
        self.rng = random.Random()
        if self.main_cfg.get("rng_seed") is not None:
            self.rng.seed(self.main_cfg["rng_seed"])

        self.accent_mapper = AccentMapper(
            self.main_cfg.get("accent_threshold", 0.6),
            tuple(self.main_cfg.get("ghost_density_range", [0.3, 0.8])),
        )
        self.ghost_hat_on_offbeat = self.main_cfg.get("ghost_hat_on_offbeat", True)
        self.drum_brush = bool(self.main_cfg.get("drum_brush", False))

        # apply groove pretty
        global_cfg = self.main_cfg.get("global_settings", {})
        groove_path = global_cfg.get("groove_profile_path")
        self.groove_strength = float(global_cfg.get("groove_strength", 1.0))
        self.groove_profile = {}
        if groove_path:
            try:
                with open(groove_path, "r", encoding="utf-8") as f:
                    self.groove_profile = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load groove profile from {groove_path}: {e}")

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
        lib = yaml.safe_load(
            open(self.main_cfg["paths"]["rhythm_library_path"], "r", encoding="utf-8")
        )
        self.raw_pattern_lib.update(lib.get("drum_patterns", {}))

        # 最終的なパターン辞書を part_parameters に適用
        self.part_parameters = self.raw_pattern_lib

    def _choose_pattern_key(
        self,
        emotion: str | None,
        intensity: str | None,
        musical_intent: Optional[Dict[str, Any]] = None,
    ) -> str:
        emo = (emotion or "default").lower()
        inten = (intensity or "medium").lower()
        bucket = EMO_TO_BUCKET.get(emo, "groove")
        base_key = BUCKET_INT_TO_PATTERN.get((bucket, inten), "groove_pocket")
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
        self.heatmap_resolution = (
            (self.main_cfg or {}).get("heatmap_resolution")
            or self.global_settings.get("heatmap_resolution", RESOLUTION)
        )
        self.heatmap_threshold = (
            (self.main_cfg or {}).get("heatmap_threshold")
            or self.global_settings.get("heatmap_threshold", 0.5)
        )

        if section_data and section_data.get("expression_details"):
            expr = section_data["expression_details"]
            key = (expr.get("emotion_bucket"), expr.get("intensity"))
            mapped = EMOTION_INTENSITY_LUT.get(key)
            if mapped:
                section_data.setdefault("part_params", {}).setdefault(self.part_name, {})[
                    "rhythm_key"
                ] = mapped

        part = super().compose(
            section_data=section_data,
            overrides_root=overrides_root,
            groove_profile_path=groove_profile_path,
            next_section_data=next_section_data,
            part_specific_humanize_params=part_specific_humanize_params,
            shared_tracks=shared_tracks,
        )

        if section_data:
            self.fill_inserter.insert(part, section_data)
        return part

    def _render(
        self,
        blocks: Sequence[Dict[str, Any]],
        part: stream.Part,
    ):
        ms_since_fill = 0
        bars_since_section_start = 0
        for blk_idx, blk_data in enumerate(blocks):
            log_render_prefix = f"DrumGen.Render.Blk{blk_idx+1}"  # 1-indexed for logs
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
            velocity_curve_list = resolve_velocity_curve(style_options.get("velocity_curve"))

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

                preferred_positions = [int(p) for p in style_def.get("preferred_fill_positions", []) if isinstance(p, int)]
                fill_keys = style_def.get("fill_patterns", [])
                at_section_end = blk_idx == len(blocks) - 1 and is_last_pattern_iteration_in_block
                bar_number = bars_since_section_start + 1
                if (
                    not fill_applied_this_iter
                    and fill_keys
                    and (bar_number in preferred_positions or at_section_end)
                ):
                    candidates = [
                        fk
                        for fk in fill_keys
                        if self._get_effective_pattern_def(fk).get("length_beats", pattern_unit_length_ql)
                        == pattern_unit_length_ql
                    ]
                    if candidates:
                        fill_key = self.rng.choice(candidates)
                        fill_def = self._get_effective_pattern_def(fill_key)
                        pattern_to_use_for_iteration = fill_def.get("pattern", [])
                        fill_legato = bool(fill_def.get("legato"))
                        fill_applied_this_iter = True
                        self.fill_offsets.append(offset_in_score + current_pos_within_block)
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
                                chosen_fill_pattern_list = fill_def.get(chosen_fill_key, fill_def.get("pattern"))
                                if chosen_fill_pattern_list is not None:
                                    pattern_to_use_for_iteration = chosen_fill_pattern_list
                                    fill_legato = bool(fill_def.get("legato"))
                                    fill_applied_this_iter = True
                                logger.debug(
                                    f"{log_render_prefix}: Applied scheduled fill '{chosen_fill_key}' for style '{style_key}'."
                                )
                start_bin = int((offset_in_score + current_pos_within_block) * self.heatmap_resolution)
                end_bin = int((offset_in_score + current_pos_within_block + current_pattern_iteration_ql) * self.heatmap_resolution)
                max_bin_val = 0
                for b in range(start_bin, end_bin):
                    max_bin_val = max(max_bin_val, self.heatmap.get(b % self.heatmap_resolution, 0))
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
    ):
        log_apply_prefix = f"DrumGen.ApplyPattern"
        beat_len_ql = (
            current_pattern_ts.beatDuration.quarterLength if current_pattern_ts else 1.0
        )

        prev_note: Optional[note.Note] = None
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
            inst_name = MISSING_DRUM_MAP_FALLBACK.get(inst_name.lower(), inst_name.lower())
            inst_name = DRUM_ALIAS.get(inst_name, inst_name).lower()
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
            if abs(swing_ratio - 0.5) > 1e-3:
                rel_offset_in_pattern = self._swing(
                    rel_offset_in_pattern, swing_ratio, beat_len_ql, swing_type
                )
            if rel_offset_in_pattern >= current_bar_actual_len_ql - (
                MIN_NOTE_DURATION_QL / 16.0
            ):
                continue

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

            if ev_def.get("type") == "ghost":
                final_event_velocity = int(pattern_base_velocity * 0.2)
                clipped_duration_ql = min(clipped_duration_ql, 0.1)
            else:
                final_event_velocity = safe_get(
                ev_def,
                "velocity",
                default=int(
                    pattern_base_velocity
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
            final_event_velocity = max(1, min(127, final_event_velocity))

            final_insert_offset_in_score = bar_start_abs_offset + rel_offset_in_pattern
            bin_idx = int((final_insert_offset_in_score * self.heatmap_resolution)) % self.heatmap_resolution
            bin_count = self.heatmap.get(bin_idx, 0)
            rel = bin_count / self.max_heatmap_value if self.max_heatmap_value else 0

            if inst_name in {"ghost", "ghost_hat"} and bin_count >= self.heatmap_threshold:
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
                final_event_velocity = self.accent_mapper.accent(rel, final_event_velocity)

            layer_idx = ev_def.get("velocity_layer")
            if velocity_curve and layer_idx is not None:
                try:
                    idx = int(layer_idx)
                    if 0 <= idx < len(velocity_curve):
                        final_event_velocity = int(final_event_velocity * velocity_curve[idx])
                except (TypeError, ValueError):
                    pass

            final_event_velocity = max(1, min(127, int(final_event_velocity * velocity_scale)))

            if ev_def.get("type") == "flam":
                midi_pitch = self.gm_pitch_map.get(inst_name)
                if midi_pitch is not None:
                    self._insert_flam(
                        part, final_insert_offset_in_score, midi_pitch, final_event_velocity
                    )
                continue

            drum_hit_note = self._make_hit(
                inst_name, final_event_velocity, clipped_duration_ql
            )
            if not drum_hit_note:
                continue

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
            if legato and prev_note is not None:
                prev_note.tie = tie.Tie('hold')
            part.insert(final_insert_offset_in_score, drum_hit_note)
            prev_note = drum_hit_note

    def _swing(
        self,
        rel_offset: float,
        swing_ratio: float,
        beat_len_ql: float,
        swing_type: str = "eighth",
    ) -> float:
        # (前回と同様)
        if abs(swing_ratio - 0.5) < 1e-3 or beat_len_ql <= 0:
            return rel_offset
        subdivision_duration_ql: float
        if swing_type == "eighth":
            subdivision_duration_ql = beat_len_ql / 2.0
        elif swing_type == "sixteenth":
            subdivision_duration_ql = beat_len_ql / 4.0
        else:
            logger.warning(
                f"DrumGen _swing: Unsupported swing_type '{swing_type}'. No swing."
            )
            return rel_offset
        if subdivision_duration_ql <= 0:
            return rel_offset
        effective_beat_for_swing_pair_ql = subdivision_duration_ql * 2.0
        beat_num_in_bar_for_swing_pair = math.floor(
            rel_offset / effective_beat_for_swing_pair_ql
        )
        offset_within_effective_beat = rel_offset - (
            beat_num_in_bar_for_swing_pair * effective_beat_for_swing_pair_ql)
        epsilon = subdivision_duration_ql * 0.1
        if abs(offset_within_effective_beat - subdivision_duration_ql) < epsilon:
            new_offset_within_effective_beat = (
                effective_beat_for_swing_pair_ql * swing_ratio
            )
            swung_rel_offset = (
                beat_num_in_bar_for_swing_pair * effective_beat_for_swing_pair_ql
            ) + new_offset_within_effective_beat
            return swung_rel_offset
        return rel_offset

    def _make_hit(self, name: str, vel: int, ql: float) -> Optional[note.Note]:
        # (前回と同様)
        mapped_name = name.lower().replace(" ", "_").replace("-", "_")
        if self.drum_brush and mapped_name in BRUSH_MAP:
            mapped_name = BRUSH_MAP[mapped_name]
        actual_name_for_midi = GHOST_ALIAS.get(mapped_name, mapped_name)
        midi_pitch_val = self.gm_pitch_map.get(actual_name_for_midi)
        if midi_pitch_val is None:
            logger.warning(
                f"DrumGen _make_hit: Unknown drum sound '{name}' (mapped to '{actual_name_for_midi}'). MIDI mapping not found. Skipping."
            )
            return None
        n = note.Note()
        n.pitch = pitch.Pitch(midi=midi_pitch_val)
        n.duration = m21duration.Duration(quarterLength=max(MIN_NOTE_DURATION_QL / 8.0, ql))
        n.volume = m21volume.Volume(velocity=max(1, min(127, vel)))
        n.offset = 0.0
        return n

    def _insert_flam(
        self, part: stream.Part, offset: float, midi_pitch: int, velocity: int
    ) -> None:
        """Insert a flam consisting of a grace note before the main hit."""
        grace_offset = (30.0 / 1000.0) * (self.global_tempo / 60.0)
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
                        if "drum_patterns" in doc and isinstance(doc["drum_patterns"], dict):
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
        """単一のコードブロックに対してドラムパートを生成する"""
        part = stream.Part(id=self.part_name)
        part.insert(0, self.default_instrument)

        drum_params = section_data.get("part_params", {}).get(self.part_name, {})
        musical_intent = section_data.get("musical_intent", {})

        style_key = self._resolve_style_key(
            musical_intent,
            drum_params,
            section_data,
        )
        pattern_def = self.part_parameters.get(style_key)

        if not pattern_def or not pattern_def.get("pattern"):
            logger.warning(
                f"Drum pattern for key '{style_key}' is empty or not found. Skipping block."
            )
            part.append(note.Rest(quarterLength=section_data.get("q_length", 4.0)))
            return part

        block_duration = section_data.get("q_length", 4.0)
        pattern_events = pattern_def.get("pattern", [])
        # apply groove pretty
        adjusted_events = []
        resolution = RESOLUTION
        for ev in pattern_events:
            ev_copy = ev.copy()
            bin_idx = int(float(ev_copy.get("offset", 0.0)) * resolution)
            groove_offset = self.groove_profile.get(str(bin_idx), 0)
            ev_copy["offset"] = float(ev_copy.get("offset", 0.0)) + groove_offset * self.groove_strength
            adjusted_events.append(ev_copy)
        pattern_events = adjusted_events
        pattern_ref_duration = float(pattern_def.get("length_beats", 4.0))
        time_scale_factor = (
            block_duration / pattern_ref_duration if pattern_ref_duration > 0 else 1.0
        )
        base_velocity = drum_params.get("velocity", 80)

        for event in pattern_events:
            event_offset = float(event.get("offset", 0.0))
            event_dur = float(event.get("duration", 0.1))
            inst_name = event.get("instrument")
            vel_factor = float(event.get("velocity_factor", 1.0))

            if not inst_name:
                continue
            gm_name, midi_pitch = self.drum_map.get(inst_name, (None, None))
            if midi_pitch is None:
                logger.warning("Unknown drum label %s", inst_name)
                continue
            inst_name = gm_name
            inst_name = MISSING_DRUM_MAP_FALLBACK.get(inst_name.lower(), inst_name.lower())
            inst_name = DRUM_ALIAS.get(inst_name, inst_name).lower()
            
            final_offset = event_offset * time_scale_factor
            if final_offset >= block_duration:
                continue

            final_dur = event_dur * time_scale_factor
            final_velocity = max(1, min(127, int(base_velocity * vel_factor)))

            bin_idx = int((final_offset * self.heatmap_resolution)) % self.heatmap_resolution
            heat_val = self.heatmap.get(bin_idx, 0)
            rel = heat_val / self.max_heatmap_value if self.max_heatmap_value else 0
            final_velocity = self.accent_mapper.get_velocity(rel, final_velocity)
            if "chh" in inst_name.lower():
                if self.rng.random() > self.accent_mapper.ghost_density(rel):
                    continue

            midi_pitch = self.gm_pitch_map.get(inst_name.lower())

            if midi_pitch:
                if inst_name.lower() in {"kick", "bd", "acoustic_bass_drum"}:
                    abs_off = section_data.get("absolute_offset", 0.0) + final_offset
                    self.kick_offsets.append(abs_off)
                n = note.Note()
                n.pitch = pitch.Pitch(midi=midi_pitch)
                n.duration.quarterLength = final_dur
                n.volume = m21volume.Volume(velocity=final_velocity)
                part.insert(final_offset, n)
            else:
                if self.strict_drum_map:
                    raise KeyError(f"Unknown drum instrument: '{inst_name}'")
                if inst_name not in self._warned_missing_drum_map:
                    logger.warning(f"Unknown drum instrument: '{inst_name}'")
                    self._warned_missing_drum_map.add(inst_name)

        profile_name = (
            (self.main_cfg or {}).get("humanize_profile")
            or section_data.get("humanize_profile")
            or self.global_settings.get("humanize_profile")
        )
        if profile_name:
            humanizer.apply(part, profile_name)

        return part

    def get_kick_offsets(self) -> List[float]:
        return list(self.kick_offsets)

    def get_fill_offsets(self) -> List[float]:
        return list(self.fill_offsets)


# --- END OF FILE generator/drum_generator.py ---

