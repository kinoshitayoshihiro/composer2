# --- START OF FILE generator/drum_generator.py (GM_DRUM_MAP確認・safe_get適用強化版) ---
from __future__ import annotations

import logging, random, math, copy
from typing import Any, Dict, List, Optional, Sequence, Union, Set

from music21 import (
    stream,
    note,
    pitch,
    volume as m21volume,
    duration as m21dur,
    tempo,
    meter,
    instrument as m21instrument,
)

# safe_get をインポート
try:
    from utilities.safe_get import safe_get
except ImportError:
    logger_fallback_safe_get_drum = logging.getLogger(__name__ + ".fallback_safe_get_drum")
    logger_fallback_safe_get_drum.error("DrumGen: CRITICAL - Could not import safe_get from utilities. Fallback will be basic .get().")
    def safe_get(data, key_path, default=None, cast_to=None, log_name="dummy_safe_get"): # ダミー
        val = data.get(key_path.split('.')[0])
        if val is None: return default
        if cast_to:
            try: return cast_to(val)
            except: return default
        return val

try:
    from utilities.override_loader import get_part_override
    from utilities.core_music_utils import MIN_NOTE_DURATION_QL, get_time_signature_object
    from utilities.humanizer import apply_humanization_to_element
except ImportError:
    logger_fallback_utils_dg = logging.getLogger(__name__ + ".fallback_utils_dg")
    logger_fallback_utils_dg.warning("DrumGen: Could not import from utilities. Using fallbacks for core utils.")
    MIN_NOTE_DURATION_QL = 0.125
    def get_time_signature_object(ts_str: Optional[str]) -> meter.TimeSignature:
        if not ts_str: ts_str = "4/4"
        try: return meter.TimeSignature(ts_str)
        except Exception: return meter.TimeSignature("4/4")
    def apply_humanization_to_element(element, template_name=None, custom_params=None): return element
    class DummyPartOverride:
        model_config: Dict[str, Any] = {}
        model_fields: Dict[str, Any] = {}

        def model_dump(self, exclude_unset: bool = True) -> Dict[str, Any]:
            """未設定フィールドを除外して辞書化（ダミー実装）。"""
            return {}
    def get_part_override(overrides, section, part, cli_override=None) -> DummyPartOverride: return DummyPartOverride()


logger = logging.getLogger("modular_composer.drum_generator")

GM_DRUM_MAP: Dict[str, int] = {
    "kick": 36, "bd": 36, "acoustic_bass_drum": 35,
    "snare": 38, "sd": 38, "acoustic_snare": 38, "electric_snare": 40,
    "closed_hi_hat": 42, "chh": 42, "closed_hat": 42,
    "pedal_hi_hat": 44, "phh": 44,
    "open_hi_hat": 46, "ohh": 46, "open_hat": 46,
    "crash_cymbal_1": 49, "crash": 49, "crash_cymbal_2": 57,
    "crash_cymbal_soft_swell": 49, # ★★★ 確認: Crash Cymbal 1 (49) にマッピング ★★★
    "ride_cymbal_1": 51, "ride": 51, "ride_cymbal_2": 59, "ride_bell": 53,
    "hand_clap": 39, "claps": 39, "side_stick": 37, "rim": 37, "rim_shot": 37,
    "low_floor_tom": 41, "tom_floor_low": 41, "high_floor_tom": 43, "tom_floor_high": 43,
    "low_tom": 45, "tom_low": 45, "low_mid_tom": 47, "tom_mid_low": 47, "tom_mid": 47,
    "high_mid_tom": 48, "tom_mid_high": 48, "tom1": 48, "high_tom": 50, "tom_hi": 50,
    "tom2": 47, "tom3": 45, "hat": 42, "stick": 31, "tambourine": 54,
    "splash": 55, "splash_cymbal": 55, "cowbell": 56, "china": 52, "china_cymbal": 52,
    "shaker": 82, "cabasa": 69, "triangle": 81,
    "wood_block_high": 76, "high_wood_block": 76, "wood_block_low": 77, "low_wood_block": 77,
    "guiro_short": 73, "short_guiro": 73, "guiro_long": 74, "long_guiro": 74, "claves": 75,
    "bongo_high": 60, "high_bongo": 60, "bongo_low": 61, "low_bongo": 61,
    "conga_open": 62, "mute_high_conga": 62, "conga_slap": 63, "open_high_conga": 63,
    "timbale_high": 65, "high_timbale": 65, "timbale_low": 66, "low_timbale": 66,
    "agogo_high": 67, "high_agogo": 67, "agogo_low": 68, "low_agogo": 68,
    "ghost_snare": 38,
}
GHOST_ALIAS: Dict[str, str] = {"ghost_snare": "snare", "gs": "snare"}

EMOTION_TO_BUCKET: Dict[str, str] = { # (前回と同様)
    "quiet_pain_and_nascent_strength": "ballad_soft", "self_reproach_regret_deep_sadness": "ballad_soft",
    "memory_unresolved_feelings_silence": "ballad_soft", "reflective_transition_instrumental_passage": "ballad_soft",
    "deep_regret_gratitude_and_realization": "groove_mid", "supported_light_longing_for_rebirth": "groove_mid",
    "wavering_heart_gratitude_chosen_strength": "groove_mid", "hope_dawn_light_gentle_guidance": "groove_mid",
    "nature_memory_floating_sensation_forgiveness": "groove_mid", "acceptance_of_love_and_pain_hopeful_belief": "anthem_high",
    "trial_cry_prayer_unbreakable_heart": "anthem_high", "reaffirmed_strength_of_love_positive_determination": "anthem_high",
    "future_cooperation_our_path_final_resolve_and_liberation": "anthem_high", "default": "groove_mid", "neutral": "groove_mid",
}
BUCKET_INTENSITY_TO_STYLE: Dict[str, Dict[str, str]] = { # (前回と同様)
    "ballad_soft": {"low": "no_drums_or_gentle_cymbal_swell", "medium_low": "ballad_soft_kick_snare_8th_hat", "medium": "ballad_soft_kick_snare_8th_hat", "medium_high": "rock_ballad_build_up_8th_hat", "high": "rock_ballad_build_up_8th_hat", "default": "ballad_soft_kick_snare_8th_hat"},
    "groove_mid": {"low": "ballad_soft_kick_snare_8th_hat", "medium_low": "rock_ballad_build_up_8th_hat", "medium": "rock_ballad_build_up_8th_hat", "medium_high": "anthem_rock_chorus_16th_hat", "high": "anthem_rock_chorus_16th_hat", "default": "rock_ballad_build_up_8th_hat"},
    "anthem_high": {"low": "rock_ballad_build_up_8th_hat", "medium_low": "anthem_rock_chorus_16th_hat", "medium": "anthem_rock_chorus_16th_hat", "medium_high": "anthem_rock_chorus_16th_hat", "high": "anthem_rock_chorus_16th_hat", "default": "anthem_rock_chorus_16th_hat"},
    "default_fallback_bucket": {"low": "no_drums", "medium_low": "default_drum_pattern", "medium": "default_drum_pattern", "medium_high": "default_drum_pattern", "high": "default_drum_pattern", "default": "default_drum_pattern"},
}

def _resolve_style(emotion: str, intensity: str, pattern_lib: Dict[str, Any]) -> str:
    # (前回と同様)
    bucket = EMOTION_TO_BUCKET.get(emotion.lower(), "default_fallback_bucket")
    style_map_for_bucket = BUCKET_INTENSITY_TO_STYLE.get(bucket)
    if not style_map_for_bucket: logger.error(f"DrumGen _resolve_style: CRITICAL - Bucket '{bucket}' not defined. Using 'default_drum_pattern'."); return "default_drum_pattern"
    resolved_style = style_map_for_bucket.get(intensity.lower())
    if not resolved_style: resolved_style = style_map_for_bucket.get("default", "default_drum_pattern")
    if resolved_style not in pattern_lib:
        logger.warning(f"DrumGen _resolve_style: Resolved style '{resolved_style}' (E:'{emotion}',I:'{intensity}') not in pattern_lib. Falling back to 'default_drum_pattern'.")
        if "default_drum_pattern" not in pattern_lib: logger.error("DrumGen _resolve_style: CRITICAL - Fallback 'default_drum_pattern' also not in pattern_lib. Returning 'no_drums'."); return "no_drums"
        return "default_drum_pattern"
    return resolved_style

class DrumGenerator:
    def __init__(self, lib: Optional[Dict[str, Dict[str, Any]]] = None, tempo_bpm: int = 120, time_sig: str = "4/4"):
        # (初期化ロジックは前回と同様)
        self.raw_pattern_lib = copy.deepcopy(lib) if lib is not None else {}
        self.pattern_lib_cache: Dict[str, Dict[str, Any]] = {}
        self.rng = random.Random()
        logger.info(f"DrumGen __init__: Initialized with {len(self.raw_pattern_lib)} raw drum patterns.")
        core_defaults = {
            "default_drum_pattern": {"description": "Default fallback pattern", "pattern": [{"offset": 0.0, "duration": 0.125, "instrument": "kick", "velocity_factor": 1.0}, {"offset": 2.0, "duration": 0.125, "instrument": "snare", "velocity_factor": 0.9}], "time_signature": "4/4", "swing": 0.5, "length_beats": 4.0, "fill_ins": {}, "velocity_base": 80},
            "no_drums": {"description": "Silence", "pattern": [], "time_signature": "4/4", "swing": 0.5, "length_beats": 4.0, "fill_ins": {}, "velocity_base": 0},
            "no_drums_or_gentle_cymbal_swell": {"description": "Placeholder: Gentle cymbal swell or silence", "pattern": [], "velocity_base": 50}, "ballad_soft_kick_snare_8th_hat": {"description": "Placeholder: Soft ballad beat", "pattern": [], "velocity_base": 70}, "rock_ballad_build_up_8th_hat": {"description": "Placeholder: Rock ballad build-up", "pattern": [], "velocity_base": 85}, "anthem_rock_chorus_16th_hat": {"description": "Placeholder: Anthem rock chorus", "pattern": [], "velocity_base": 100}, "no_drums_or_sparse_cymbal": {"description": "Placeholder: Sparse cymbal or silence", "pattern": [], "velocity_base": 40}, "no_drums_or_sparse_chimes": {"description": "Placeholder: Sparse chimes or silence", "pattern": [], "velocity_base": 45},
        }
        for k, v_def_template in core_defaults.items():
            if k not in self.raw_pattern_lib:
                placeholder_def = {"description": v_def_template.get("description", f"Placeholder for '{k}'."), "pattern": v_def_template.get("pattern", []), "time_signature": v_def_template.get("time_signature", "4/4"), "swing": v_def_template.get("swing", 0.5), "length_beats": v_def_template.get("length_beats", 4.0), "fill_ins": v_def_template.get("fill_ins", {}), "velocity_base": v_def_template.get("velocity_base", 70)}
                self.raw_pattern_lib[k] = placeholder_def; logger.info(f"DrumGen __init__: Added/updated placeholder for style '{k}'.")
        all_referenced_styles_in_luts: Set[str] = set()
        for bucket_styles in BUCKET_INTENSITY_TO_STYLE.values(): all_referenced_styles_in_luts.update(bucket_styles.values())
        for style_key in all_referenced_styles_in_luts:
            if style_key not in self.raw_pattern_lib:
                self.raw_pattern_lib[style_key] = {"description": f"Auto-added placeholder for undefined style '{style_key}'.", "pattern": [], "time_signature": "4/4", "swing": 0.5, "length_beats": 4.0, "fill_ins": {}, "velocity_base": 70}
                logger.info(f"DrumGen __init__: Added silent placeholder for undefined style '{style_key}' (from LUT).")
        self.global_tempo = tempo_bpm; self.global_time_signature_str = time_sig
        self.global_ts = get_time_signature_object(time_sig)
        if not self.global_ts: logger.warning(f"DrumGen __init__: Failed to parse global time_sig '{time_sig}'. Defaulting to 4/4."); self.global_ts = meter.TimeSignature("4/4")
        self.instrument = m21instrument.Percussion()
        if hasattr(self.instrument, 'midiChannel'): self.instrument.midiChannel = 9

    def _get_effective_pattern_def(self, style_key: str, visited: Optional[Set[str]] = None) -> Dict[str, Any]:
        # (前回と同様の継承解決ロジック)
        if visited is None: visited = set()
        if style_key in visited:
            logger.error(f"DrumGen: Circular inheritance for '{style_key}'. Returning 'default_drum_pattern'.")
            default_p_data = self.pattern_lib_cache.get("default_drum_pattern") or self.raw_pattern_lib.get("default_drum_pattern", {})
            return copy.deepcopy(default_p_data if default_p_data else {"pattern":[]})
        if style_key in self.pattern_lib_cache: return copy.deepcopy(self.pattern_lib_cache[style_key])
        pattern_def_original = self.raw_pattern_lib.get(style_key)
        if not pattern_def_original:
            logger.warning(f"DrumGen: Style key '{style_key}' not found. Falling back to 'default_drum_pattern'.")
            default_p = self.raw_pattern_lib.get("default_drum_pattern")
            if not default_p: logger.error("DrumGen: CRITICAL - 'default_drum_pattern' missing. Returning minimal empty."); return {"description": "Minimal Empty (Critical Fallback)", "pattern": [], "time_signature": "4/4", "swing": 0.5, "length_beats": 4.0, "fill_ins": {}, "velocity_base": 70}
            self.pattern_lib_cache[style_key] = copy.deepcopy(default_p); return default_p
        pattern_def = copy.deepcopy(pattern_def_original)
        inherit_key = pattern_def.get("inherit")
        if inherit_key and isinstance(inherit_key, str):
            logger.debug(f"DrumGen: Pattern '{style_key}' inherits '{inherit_key}'. Resolving...")
            visited.add(style_key); base_def = self._get_effective_pattern_def(inherit_key, visited); visited.remove(style_key)
            merged_def = base_def.copy()
            if "pattern" in pattern_def: merged_def["pattern"] = pattern_def["pattern"]
            base_fills = merged_def.get("fill_ins", {}); current_fills = pattern_def.get("fill_ins", {})
            if isinstance(base_fills, dict) and isinstance(current_fills, dict): merged_fills = base_fills.copy(); merged_fills.update(current_fills); merged_def["fill_ins"] = merged_fills
            elif current_fills is not None: merged_def["fill_ins"] = current_fills
            for key, value in pattern_def.items():
                if key not in ["inherit", "pattern", "fill_ins"]: merged_def[key] = value
            pattern_def = merged_def
        pattern_def.setdefault("time_signature", self.global_time_signature_str)
        pattern_def.setdefault("swing", 0.5)
        pattern_def.setdefault("length_beats", get_time_signature_object(pattern_def["time_signature"]).barDuration.quarterLength if get_time_signature_object(pattern_def["time_signature"]) else 4.0)
        pattern_def.setdefault("pattern", []); pattern_def.setdefault("fill_ins", {}); pattern_def.setdefault("velocity_base", 80)
        self.pattern_lib_cache[style_key] = copy.deepcopy(pattern_def); return pattern_def

    def compose(self, blocks: List[Dict[str, Any]], overrides: Optional[Any]=None) -> stream.Part:
        # (前回と同様の compose メソッドの冒頭部分)
        part = stream.Part(id="Drums"); part.insert(0, copy.deepcopy(self.instrument))
        part.insert(0, tempo.MetronomeMark(number=self.global_tempo))
        current_ts_obj = self.global_ts if self.global_ts else meter.TimeSignature("4/4")
        part.insert(0, copy.deepcopy(current_ts_obj))
        if not blocks: return part
        logger.info(f"DrumGen compose: Starting for {len(blocks)} blocks.")
        resolved_blocks = []
        for blk_idx, blk_data_original in enumerate(blocks):
            blk = copy.deepcopy(blk_data_original); current_section_name = blk.get("section_name", f"UnnamedSection_{blk_idx}")
            part_specific_overrides_model: Optional[DummyPartOverride] = None
            if overrides: part_specific_overrides_model = get_part_override(overrides, current_section_name, "drums")
            blk.setdefault("part_params", {}).setdefault("drums", {})
            drum_params_from_chordmap = blk["part_params"]["drums"]; final_drum_params = drum_params_from_chordmap.copy()
            if part_specific_overrides_model and hasattr(part_specific_overrides_model, 'model_dump'):
                override_dict = part_specific_overrides_model.model_dump(exclude_unset=True)
                chordmap_options = final_drum_params.get("options", {}); override_options = override_dict.pop("options", None)
                if isinstance(chordmap_options, dict) and isinstance(override_options, dict): merged_options = chordmap_options.copy(); merged_options.update(override_options); final_drum_params["options"] = merged_options
                elif isinstance(override_options, dict) and override_options: final_drum_params["options"] = override_options
                final_drum_params.update(override_dict)
            blk["part_params"]["drums"] = final_drum_params
            emo = blk.get("musical_intent", {}).get("emotion", "default").lower(); inten = blk.get("musical_intent", {}).get("intensity", "medium").lower()
            style_key_from_override = final_drum_params.get("rhythm_key")
            if style_key_from_override and style_key_from_override in self.raw_pattern_lib: final_style_key = style_key_from_override; logger.debug(f"DrumGen compose: Blk {blk_idx+1} using style '{final_style_key}' from overrides.")
            else:
                explicit_style_key_chordmap = final_drum_params.get("drum_style_key", final_drum_params.get("style_key"))
                if explicit_style_key_chordmap and explicit_style_key_chordmap in self.raw_pattern_lib: final_style_key = explicit_style_key_chordmap; logger.debug(f"DrumGen compose: Blk {blk_idx+1} using explicit style '{final_style_key}' from chordmap.")
                else: final_style_key = _resolve_style(emo, inten, self.raw_pattern_lib); logger.debug(f"DrumGen compose: Blk {blk_idx+1} (E:'{emo}',I:'{inten}') using auto-resolved style '{final_style_key}'")
            blk["part_params"]["drums"]["final_style_key_for_render"] = final_style_key
            resolved_blocks.append(blk)
        self._render(resolved_blocks, part)
        logger.info(f"DrumGen compose: Finished. Part has {len(list(part.flatten().notesAndRests))} elements.")
        return part

    def _render(self, blocks: Sequence[Dict[str, Any]], part: stream.Part):
        ms_since_fill = 0
        for blk_idx, blk_data in enumerate(blocks):
            log_render_prefix = f"DrumGen.Render.Blk{blk_idx+1}" # 1-indexed for logs
            drums_params = blk_data.get("part_params", {}).get("drums", {})
            style_key = drums_params.get("final_style_key_for_render", "default_drum_pattern")
            style_def = self._get_effective_pattern_def(style_key)
            if not style_def: logger.error(f"{log_render_prefix}: CRITICAL - No style_def for '{style_key}'. Skipping."); continue

            # --- base_vel の取得 (safe_get を使用) ---
            base_vel = safe_get(drums_params, "velocity", 
                                default=safe_get(drums_params, "drum_base_velocity", 
                                                 default=safe_get(style_def, "velocity_base", 
                                                                  default=80, cast_to=int, log_name=f"{log_render_prefix}.VelStyleDef"), 
                                                 cast_to=int, log_name=f"{log_render_prefix}.VelDrumBaseParam"), 
                                cast_to=int, log_name=f"{log_render_prefix}.VelParam")
            base_vel = max(1, min(127, base_vel))
            # --- ここまで base_vel ---

            pat_events: List[Dict[str, Any]] = style_def.get("pattern", [])
            pat_ts_str = style_def.get("time_signature", self.global_time_signature_str)
            pat_ts = get_time_signature_object(pat_ts_str)
            if not pat_ts: pat_ts = self.global_ts

            pattern_unit_length_ql = safe_get(style_def, "length_beats", default=pat_ts.barDuration.quarterLength if pat_ts else 4.0, cast_to=float, log_name=f"{log_render_prefix}.PatternLen")
            if pattern_unit_length_ql <= 0: logger.warning(f"{log_render_prefix}: Pattern '{style_key}' invalid length {pattern_unit_length_ql}. Defaulting to 4.0"); pattern_unit_length_ql = 4.0

            swing_setting = style_def.get("swing", 0.5)
            swing_type = "eighth"; swing_ratio_val = 0.5
            if isinstance(swing_setting, dict):
                swing_type = swing_setting.get("type", "eighth").lower()
                swing_ratio_val = safe_get(swing_setting, "ratio", default=0.5, cast_to=float, log_name=f"{log_render_prefix}.SwingRatio")
            elif isinstance(swing_setting, (float, int)): swing_ratio_val = float(swing_setting)

            fills = style_def.get("fill_ins", {})
            
            # --- オフセットとデュレーション (safe_get を使用) ---
            default_block_dur = pattern_unit_length_ql if pattern_unit_length_ql > 0 else 4.0
            offset_in_score = safe_get(blk_data, "humanized_offset_beats", default=safe_get(blk_data, "absolute_offset", default=safe_get(blk_data, "offset", default=0.0, cast_to=float, log_name=f"{log_render_prefix}.OffsetFallback3"), cast_to=float, log_name=f"{log_render_prefix}.OffsetFallback2"), cast_to=float, log_name=f"{log_render_prefix}.HumOffset")
            remaining_ql_in_block = safe_get(blk_data, "humanized_duration_beats", default=safe_get(blk_data, "q_length", default=default_block_dur, cast_to=float, log_name=f"{log_render_prefix}.QLFallback"), cast_to=float, log_name=f"{log_render_prefix}.HumDur")
            if remaining_ql_in_block <= 0: logger.warning(f"{log_render_prefix}: Non-positive duration {remaining_ql_in_block} (raw: {blk_data.get('humanized_duration_beats', blk_data.get('q_length'))}). Using {default_block_dur}ql."); remaining_ql_in_block = default_block_dur
            # --- ここまでオフセットとデュレーション ---

            if blk_data.get("is_first_in_section", False) and blk_idx > 0: ms_since_fill = 0
            current_pos_within_block = 0.0
            while remaining_ql_in_block > MIN_NOTE_DURATION_QL / 8.0:
                # (フィルインロジック、パターンの適用は前回と同様、base_vel を _apply_pattern に渡す)
                current_pattern_iteration_ql = min(pattern_unit_length_ql, remaining_ql_in_block)
                if current_pattern_iteration_ql < MIN_NOTE_DURATION_QL / 4.0: break
                is_last_pattern_iteration_in_block = remaining_ql_in_block <= pattern_unit_length_ql + (MIN_NOTE_DURATION_QL / 8.0)
                pattern_to_use_for_iteration = pat_events; fill_applied_this_iter = False
                override_fill_key = drums_params.get("fill_override", drums_params.get("drum_fill_key_override"))
                if is_last_pattern_iteration_in_block and override_fill_key:
                    chosen_fill_pattern_list = fills.get(override_fill_key)
                    if chosen_fill_pattern_list is not None: pattern_to_use_for_iteration = chosen_fill_pattern_list; fill_applied_this_iter = True; logger.debug(f"{log_render_prefix}: Applied override fill '{override_fill_key}' for style '{style_key}'.")
                    else: logger.warning(f"{log_render_prefix}: Override fill key '{override_fill_key}' not in fills for '{style_key}'.")
                fill_interval_bars = safe_get(drums_params, "drum_fill_interval_bars", default=0, cast_to=int, log_name=f"{log_render_prefix}.FillInterval")
                if not fill_applied_this_iter and is_last_pattern_iteration_in_block and fill_interval_bars > 0:
                    if (ms_since_fill + 1) >= fill_interval_bars :
                        fill_keys_from_params = drums_params.get("drum_fill_keys", []); possible_fills_for_style = [fk for fk in fill_keys_from_params if fk in fills]
                        if possible_fills_for_style:
                            chosen_fill_key = self.rng.choice(possible_fills_for_style); chosen_fill_pattern_list = fills.get(chosen_fill_key)
                            if chosen_fill_pattern_list is not None: pattern_to_use_for_iteration = chosen_fill_pattern_list; fill_applied_this_iter = True; logger.debug(f"{log_render_prefix}: Applied scheduled fill '{chosen_fill_key}' for style '{style_key}'.")
                self._apply_pattern(part, pattern_to_use_for_iteration, offset_in_score + current_pos_within_block, current_pattern_iteration_ql, base_vel, swing_type, swing_ratio_val, pat_ts if pat_ts else self.global_ts, drums_params)
                if fill_applied_this_iter: ms_since_fill = 0
                else: ms_since_fill += 1
                current_pos_within_block += current_pattern_iteration_ql; remaining_ql_in_block -= current_pattern_iteration_ql

    def _apply_pattern(self, part: stream.Part, events: List[Dict[str, Any]], bar_start_abs_offset: float, current_bar_actual_len_ql: float, pattern_base_velocity: int, swing_type: str, swing_ratio: float, current_pattern_ts: meter.TimeSignature, drum_block_params: Dict[str, Any]):
        log_apply_prefix = f"DrumGen.ApplyPattern"
        beat_len_ql = current_pattern_ts.beatDuration.quarterLength if current_pattern_ts else 1.0

        for ev_idx, ev_def in enumerate(events):
            log_event_prefix = f"{log_apply_prefix}.Evt{ev_idx}"
            if self.rng.random() > safe_get(ev_def, "probability", default=1.0, cast_to=float, log_name=f"{log_event_prefix}.Prob"): continue
            inst_name = ev_def.get("instrument")
            if not inst_name: continue

            rel_offset_in_pattern = safe_get(ev_def, "offset", default=0.0, cast_to=float, log_name=f"{log_event_prefix}.Offset")
            if abs(swing_ratio - 0.5) > 1e-3 : rel_offset_in_pattern = self._swing(rel_offset_in_pattern, swing_ratio, beat_len_ql, swing_type)
            if rel_offset_in_pattern >= current_bar_actual_len_ql - (MIN_NOTE_DURATION_QL / 16.0): continue
            
            hit_duration_ql_from_def = safe_get(ev_def, "duration", default=0.125, cast_to=float, log_name=f"{log_event_prefix}.Dur")
            clipped_duration_ql = min(hit_duration_ql_from_def, current_bar_actual_len_ql - rel_offset_in_pattern)
            if clipped_duration_ql < MIN_NOTE_DURATION_QL / 8.0: continue

            final_event_velocity = safe_get(ev_def, "velocity", 
                                            default=int(pattern_base_velocity * safe_get(ev_def, "velocity_factor", default=1.0, cast_to=float, log_name=f"{log_event_prefix}.VelFactor")), 
                                            cast_to=int, log_name=f"{log_event_prefix}.VelAbs")
            final_event_velocity = max(1, min(127, final_event_velocity))

            drum_hit_note = self._make_hit(inst_name, final_event_velocity, clipped_duration_ql)
            if not drum_hit_note: continue

            # (ヒューマナイズ処理は前回と同様)
            humanize_this_hit = False; humanize_template_for_hit = "drum_tight"; humanize_custom_for_hit = {}
            event_humanize_setting = ev_def.get("humanize")
            if isinstance(event_humanize_setting, bool): humanize_this_hit = event_humanize_setting
            elif isinstance(event_humanize_setting, str): humanize_this_hit = True; humanize_template_for_hit = event_humanize_setting
            elif isinstance(event_humanize_setting, dict): humanize_this_hit = True; humanize_template_for_hit = event_humanize_setting.get("template_name", humanize_template_for_hit); humanize_custom_for_hit = event_humanize_setting.get("custom_params", {})
            else:
                if drum_block_params.get("humanize_opt", False): humanize_this_hit = True; humanize_template_for_hit = drum_block_params.get("template_name", "drum_tight"); humanize_custom_for_hit = drum_block_params.get("custom_params", {})
            time_delta_from_humanizer = 0.0
            if humanize_this_hit:
                original_hit_offset_before_humanize = drum_hit_note.offset
                drum_hit_note = apply_humanization_to_element(drum_hit_note, template_name=humanize_template_for_hit, custom_params=humanize_custom_for_hit) # human_custom_for_hit -> humanize_custom_for_hit
            final_insert_offset_in_score = bar_start_abs_offset + rel_offset_in_pattern + time_delta_from_humanizer
            drum_hit_note.offset = 0.0
            part.insert(final_insert_offset_in_score, drum_hit_note)

    def _swing(self, rel_offset: float, swing_ratio: float, beat_len_ql: float, swing_type: str = "eighth") -> float:
        # (前回と同様)
        if abs(swing_ratio - 0.5) < 1e-3 or beat_len_ql <= 0: return rel_offset
        subdivision_duration_ql: float
        if swing_type == "eighth": subdivision_duration_ql = beat_len_ql / 2.0
        elif swing_type == "sixteenth": subdivision_duration_ql = beat_len_ql / 4.0
        else: logger.warning(f"DrumGen _swing: Unsupported swing_type '{swing_type}'. No swing."); return rel_offset
        if subdivision_duration_ql <= 0: return rel_offset
        effective_beat_for_swing_pair_ql = subdivision_duration_ql * 2.0 
        beat_num_in_bar_for_swing_pair = math.floor(rel_offset / effective_beat_for_swing_pair_ql)
        offset_within_effective_beat = rel_offset - (beat_num_in_bar_for_swing_pair * effective_beat_for_swing_pair_ql)
        epsilon = subdivision_duration_ql * 0.1 
        if abs(offset_within_effective_beat - subdivision_duration_ql) < epsilon:
            new_offset_within_effective_beat = effective_beat_for_swing_pair_ql * swing_ratio
            swung_rel_offset = (beat_num_in_bar_for_swing_pair * effective_beat_for_swing_pair_ql) + new_offset_within_effective_beat
            return swung_rel_offset
        return rel_offset

    def _make_hit(self, name: str, vel: int, ql: float) -> Optional[note.Note]:
        # (前回と同様)
        mapped_name = name.lower().replace(" ", "_").replace("-", "_"); actual_name_for_midi = GHOST_ALIAS.get(mapped_name, mapped_name)
        midi_pitch_val = GM_DRUM_MAP.get(actual_name_for_midi)
        if midi_pitch_val is None: logger.warning(f"DrumGen _make_hit: Unknown drum sound '{name}' (mapped to '{actual_name_for_midi}'). MIDI mapping not found. Skipping."); return None
        n = note.Note(); n.pitch = pitch.Pitch(midi=midi_pitch_val)
        n.duration = m21dur.Duration(quarterLength=max(MIN_NOTE_DURATION_QL / 8.0, ql)) 
        n.volume = m21volume.Volume(velocity=max(1, min(127, vel)))
        n.offset = 0.0; return n
# --- END OF FILE ---