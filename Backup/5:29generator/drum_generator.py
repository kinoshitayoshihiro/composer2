# --- START OF FILE generator/drum_generator.py (Harugoro-OTO KOTOBA Engine - Schema & Inherit Edition) ---
from __future__ import annotations

import logging, random, math, copy # copy を追加
from typing import Any, Dict, List, Optional, Sequence, Union # Union を追加

from music21 import (
    stream, note, pitch, volume as m21volume, duration as m21dur, tempo,
    meter, instrument as m21instrument,
)

try:
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
    def apply_humanization_to_element(element, template_name=None, custom_params=None):
        return element

logger = logging.getLogger(__name__)

# GM Drum Map (o3-san feedback 반영판 / v2025‑05‑27 brush‑up 版ベースの包括的なもの)
GM_DRUM_MAP: Dict[str, int] = {
    "kick": 36, "bd": 36, "acoustic_bass_drum": 35,
    "snare": 38, "sd": 38, "acoustic_snare": 38, "electric_snare": 40,
    "closed_hi_hat": 42, "chh": 42, "closed_hat": 42,
    "pedal_hi_hat": 44, "phh": 44,
    "open_hi_hat": 46, "ohh": 46, "open_hat": 46,
    "crash_cymbal_1": 49, "crash": 49, "crash_cymbal_2": 57,
    "ride_cymbal_1": 51, "ride": 51, "ride_cymbal_2": 59,
    "ride_bell": 53,
    "hand_clap": 39, "claps": 39,
    "side_stick": 37, "rim": 37, "rim_shot": 37,
    "low_floor_tom": 41,   "tom_floor_low": 41,
    "high_floor_tom": 43,  "tom_floor_high": 43,
    "low_tom": 45,         "tom_low": 45,
    "low_mid_tom": 47,     "tom_mid_low": 47, "tom_mid": 47,
    "high_mid_tom": 48,    "tom_mid_high": 48, "tom1": 48,
    "high_tom": 50,        "tom_hi": 50,
    "tom2": 47, "tom3": 45, "hat": 42, "stick": 31,
    "tambourine": 54, "splash": 55, "splash_cymbal": 55, "cowbell": 56,
    "china": 52, "china_cymbal": 52, "shaker": 82, "cabasa": 69, "triangle": 81,
    "wood_block_high": 76, "high_wood_block": 76, "wood_block_low": 77, "low_wood_block": 77,
    "guiro_short": 73, "short_guiro": 73, "guiro_long": 74, "long_guiro": 74,
    "claves": 75, "bongo_high": 60, "high_bongo": 60, "bongo_low": 61, "low_bongo": 61,
    "conga_open": 62, "mute_high_conga": 62, "conga_slap": 63, "open_high_conga": 63,
    "timbale_high": 65, "high_timbale": 65, "timbale_low": 66, "low_timbale": 66,
    "agogo_high": 67, "high_agogo": 67, "agogo_low": 68, "low_agogo": 68,
    "ghost_snare": 38,
}
GHOST_ALIAS: Dict[str, str] = {"ghost_snare": "snare", "gs": "snare"}

# Look-up tables (前回版と同様)
EMOTION_TO_BUCKET: Dict[str, str] = {
    "quiet_pain_and_nascent_strength": "ballad_soft", "self_reproach_regret_deep_sadness": "ballad_soft",
    "memory_unresolved_feelings_silence": "ballad_soft", "reflective_transition_instrumental_passage": "ballad_soft",
    "deep_regret_gratitude_and_realization": "groove_mid", "supported_light_longing_for_rebirth": "groove_mid",
    "wavering_heart_gratitude_chosen_strength": "groove_mid", "hope_dawn_light_gentle_guidance": "groove_mid",
    "nature_memory_floating_sensation_forgiveness": "groove_mid",
    "acceptance_of_love_and_pain_hopeful_belief": "anthem_high", "trial_cry_prayer_unbreakable_heart": "anthem_high",
    "reaffirmed_strength_of_love_positive_determination": "anthem_high",
    "future_cooperation_our_path_final_resolve_and_liberation": "anthem_high",
    "default": "groove_mid", "neutral": "groove_mid"
}
BUCKET_INTENSITY_TO_STYLE: Dict[str, Dict[str, str]] = {
    "ballad_soft": {"low": "no_drums_or_gentle_cymbal_swell", "medium_low": "ballad_soft_kick_snare_8th_hat", "medium": "ballad_soft_kick_snare_8th_hat", "medium_high": "rock_ballad_build_up_8th_hat", "high": "rock_ballad_build_up_8th_hat", "default": "ballad_soft_kick_snare_8th_hat"},
    "groove_mid": {"low": "ballad_soft_kick_snare_8th_hat", "medium_low": "rock_ballad_build_up_8th_hat", "medium": "rock_ballad_build_up_8th_hat", "medium_high": "anthem_rock_chorus_16th_hat", "high": "anthem_rock_chorus_16th_hat", "default": "rock_ballad_build_up_8th_hat"},
    "anthem_high": {"low": "rock_ballad_build_up_8th_hat", "medium_low": "anthem_rock_chorus_16th_hat", "medium": "anthem_rock_chorus_16th_hat", "medium_high": "anthem_rock_chorus_16th_hat", "high": "anthem_rock_chorus_16th_hat", "default": "anthem_rock_chorus_16th_hat"},
    "default_fallback_bucket": {"low": "no_drums", "medium_low": "default_drum_pattern", "medium": "default_drum_pattern", "medium_high": "default_drum_pattern", "high": "default_drum_pattern", "default": "default_drum_pattern"}
}

def _resolve_style(emotion:str, intensity:str, pattern_lib: Dict[str, Any]) -> str:
    bucket = EMOTION_TO_BUCKET.get(emotion.lower(), "default_fallback_bucket")
    style_map_for_bucket = BUCKET_INTENSITY_TO_STYLE.get(bucket)
    if not style_map_for_bucket:
        logger.error(f"DrumGen _resolve_style: CRITICAL - Bucket '{bucket}' is not defined. Using 'default_drum_pattern'.")
        return "default_drum_pattern"
    resolved_style = style_map_for_bucket.get(intensity.lower())
    if not resolved_style:
        resolved_style = style_map_for_bucket.get("default", "default_drum_pattern")
    if resolved_style not in pattern_lib: # pattern_lib を参照して存在確認
        logger.warning(f"DrumGen _resolve_style: Resolved style '{resolved_style}' not in pattern_lib. Falling back.")
        return "default_drum_pattern"
    return resolved_style

class DrumGenerator:
    def __init__(self, lib: Optional[Dict[str,Dict[str,Any]]] = None,
                 tempo_bpm:int = 120, time_sig:str="4/4"):
        self.raw_pattern_lib = copy.deepcopy(lib) if lib is not None else {} # 元のライブラリを保持
        self.pattern_lib_cache: Dict[str, Dict[str, Any]] = {} # 継承解決済みパターンをキャッシュ
        self.rng = random.Random()
        logger.info(f"DrumGen __init__: Received raw_pattern_lib with {len(self.raw_pattern_lib)} keys.")

        core_defaults = {
            "default_drum_pattern":{"description": "Default fallback pattern", "pattern":[], "time_signature":"4/4", "swing": 0.5, "length_beats": 4.0, "fill_ins": {}},
            "no_drums":{"description": "Silence", "pattern":[],"time_signature":"4/4", "swing": 0.5, "length_beats": 4.0, "fill_ins": {}}
        }
        for k, v_def in core_defaults.items():
            if k not in self.raw_pattern_lib: # raw_pattern_lib になければ追加
                self.raw_pattern_lib[k] = v_def
                logger.info(f"DrumGen __init__: Added core default '{k}' to raw pattern library.")
        
        all_referenced_styles_in_luts = set()
        for bucket_styles in BUCKET_INTENSITY_TO_STYLE.values():
            for style_key_in_lut in bucket_styles.values():
                all_referenced_styles_in_luts.add(style_key_in_lut)
        chordmap_common_styles = ["no_drums_or_sparse_cymbal", "no_drums_or_gentle_cymbal_swell", "no_drums_or_sparse_chimes", "ballad_soft_kick_snare_8th_hat", "rock_ballad_build_up_8th_hat", "anthem_rock_chorus_16th_hat"]
        all_referenced_styles_in_luts.update(chordmap_common_styles)

        for style_key in all_referenced_styles_in_luts:
            if style_key not in self.raw_pattern_lib:
                self.raw_pattern_lib[style_key] = {
                    "description": f"Placeholder for '{style_key}' (auto-added).",
                    "time_signature": "4/4", "swing": 0.5, "length_beats": 4.0,
                    "pattern": [], "fill_ins": {}
                }
                logger.info(f"DrumGen __init__: Added silent placeholder for undefined style '{style_key}' to raw_pattern_lib.")

        self.global_tempo = tempo_bpm
        self.global_time_signature_str = time_sig
        self.global_ts = get_time_signature_object(time_sig)
        if not self.global_ts:
            logger.warning(f"DrumGen __init__: Failed to parse time_sig '{time_sig}'. Defaulting to 4/4.")
            self.global_ts = meter.TimeSignature("4/4")
        self.instrument = m21instrument.Percussion()
        if hasattr(self.instrument, "midiChannel"):
            self.instrument.midiChannel = 9

    def _get_effective_pattern_def(self, style_key: str, visited: Optional[Set[str]] = None) -> Dict[str, Any]:
        """
        Resolves pattern inheritance ("inherit" key) and returns the effective pattern definition.
        Caches resolved patterns.
        """
        if visited is None: visited = set()
        if style_key in visited:
            logger.error(f"DrumGen: Circular inheritance detected involving '{style_key}'. Returning base default.")
            return copy.deepcopy(self.raw_pattern_lib.get("default_drum_pattern", {}))
        visited.add(style_key)

        if style_key in self.pattern_lib_cache:
            return copy.deepcopy(self.pattern_lib_cache[style_key])

        pattern_def = copy.deepcopy(self.raw_pattern_lib.get(style_key))
        if not pattern_def:
            logger.warning(f"DrumGen: Style key '{style_key}' not found in raw_pattern_lib. Returning default.")
            return copy.deepcopy(self.raw_pattern_lib.get("default_drum_pattern", {}))

        inherit_key = pattern_def.get("inherit")
        if inherit_key and isinstance(inherit_key, str):
            logger.debug(f"DrumGen: Pattern '{style_key}' inherits from '{inherit_key}'.")
            base_def = self._get_effective_pattern_def(inherit_key, visited) # Recursive call
            
            # Merge: current pattern_def overrides base_def
            # For "pattern" and "fill_ins", we might want smarter merging or complete override.
            # For now, simple dictionary update (pattern_def takes precedence).
            merged_def = base_def.copy() # Start with a copy of the resolved base
            
            # Deep merge for nested dicts like 'options' or 'fill_ins' if needed in future.
            # For now, top-level keys from pattern_def will overwrite.
            # Special handling for 'pattern' list: if pattern_def has one, it overrides.
            # If pattern_def doesn't have 'pattern' but base does, base's pattern is used.
            # If pattern_def's pattern is empty and base has one, use base.
            
            current_pattern_list = pattern_def.get("pattern")
            if current_pattern_list is not None: # If current def has 'pattern' (even if empty list) it overrides base.
                 merged_def["pattern"] = current_pattern_list
            # else: base_def's pattern (if any) is kept.

            # For fill_ins, merge dictionaries (specific fills override base fills)
            base_fills = merged_def.get("fill_ins", {})
            current_fills = pattern_def.get("fill_ins", {})
            if isinstance(base_fills, dict) and isinstance(current_fills, dict):
                merged_fills = base_fills.copy()
                merged_fills.update(current_fills)
                merged_def["fill_ins"] = merged_fills
            elif current_fills is not None: # If current_fills exists (even if not dict), it overrides
                 merged_def["fill_ins"] = current_fills


            # Other keys are simply overridden by pattern_def
            for key, value in pattern_def.items():
                if key not in ["inherit", "pattern", "fill_ins"]: # Don't copy 'inherit' itself, pattern/fills handled
                    merged_def[key] = value
            pattern_def = merged_def
        
        self.pattern_lib_cache[style_key] = copy.deepcopy(pattern_def)
        visited.remove(style_key)
        return pattern_def

    def compose(self, blocks: List[Dict[str,Any]]) -> stream.Part:
        part = stream.Part(id="Drums")
        part.insert(0, self.instrument)
        part.insert(0, tempo.MetronomeMark(number=self.global_tempo))
        if self.global_ts and hasattr(self.global_ts, 'ratioString'):
            ts_to_insert = meter.TimeSignature(self.global_ts.ratioString)
        else:
            logger.warning("DrumGen compose: self.global_ts is invalid. Defaulting to 4/4.")
            ts_to_insert = meter.TimeSignature("4/4")
        part.insert(0, ts_to_insert)

        if not blocks: return part
        logger.info(f"DrumGen compose: Starting for {len(blocks)} blocks.")

        for blk_idx, blk in enumerate(blocks):
            emo = blk.get("musical_intent",{}).get("emotion","default").lower()
            inten = blk.get("musical_intent",{}).get("intensity","medium").lower()
            # _resolve_style now uses self.raw_pattern_lib for existence check
            style_key_resolved = _resolve_style(emo, inten, self.raw_pattern_lib) 
            
            blk.setdefault("part_params",{}).setdefault("drums",{})
            # chordmap.jsonからの明示的なstyle_keyも考慮
            explicit_style_key = blk["part_params"]["drums"].get("drum_style_key", blk["part_params"]["drums"].get("style_key"))
            if explicit_style_key and explicit_style_key in self.raw_pattern_lib:
                final_style_key = explicit_style_key
                logger.debug(f"DrumGen compose: Blk {blk_idx+1} using explicit style '{final_style_key}' from chordmap.")
            else:
                final_style_key = style_key_resolved
            
            blk["part_params"]["drums"]["style_key"] = final_style_key # 解決/選択されたキーを格納
            logger.debug(f"DrumGen compose: Blk {blk_idx+1} (E:'{emo}',I:'{inten}') using style '{final_style_key}'")

        self._render(blocks, part)
        logger.info(f"DrumGen compose: Finished. Part has {len(list(part.flatten().notesAndRests))} elements.")
        return part

    def _render(self, blocks:Sequence[Dict[str,Any]], part:stream.Part):
        ms_since_fill = 0
        for blk_idx, blk_data in enumerate(blocks):
            drums_params = blk_data.get("part_params", {}).get("drums", {})
            style_key = drums_params.get("style_key", "default_drum_pattern")
            
            style_def = self._get_effective_pattern_def(style_key) # <<< Use inheritance resolving getter
            if not style_def: # Should be handled by _get_effective_pattern_def returning a default
                logger.error(f"DrumGen _render: CRITICAL - _get_effective_pattern_def returned None for '{style_key}'.")
                continue

            pat_events: List[Dict[str,Any]] = style_def.get("pattern", [])
            pat_ts_str = style_def.get("time_signature", self.global_time_signature_str)
            pat_ts = get_time_signature_object(pat_ts_str)
            if not pat_ts: pat_ts = self.global_ts

            # <<< Use "length_beats" if available, otherwise calculate from TS
            pattern_unit_length_ql = float(style_def.get("length_beats", pat_ts.barDuration.quarterLength if pat_ts else 4.0))
            if pattern_unit_length_ql <=0:
                 logger.warning(f"DrumGen _render: Pattern '{style_key}' has invalid length_beats/barDuration {pattern_unit_length_ql}. Defaulting to 4.0")
                 pattern_unit_length_ql = 4.0

            # <<< Parse new swing structure
            swing_setting = style_def.get("swing", 0.5) # Default to 0.5 (straight)
            swing_type = "eighth" # Default swing type
            swing_ratio_val = 0.5
            if isinstance(swing_setting, dict):
                swing_type = swing_setting.get("type", "eighth").lower()
                swing_ratio_val = float(swing_setting.get("ratio", 0.5))
            elif isinstance(swing_setting, (float, int)):
                swing_ratio_val = float(swing_setting)
            
            fills = style_def.get("fill_ins", {})
            base_vel = int(drums_params.get("drum_base_velocity", 80))
            intensity_str = blk_data.get("musical_intent", {}).get("intensity", "medium").lower()
            if intensity_str in ["high", "medium_high"]: base_vel = min(127, base_vel + 8)
            elif intensity_str in ["low", "medium_low"]: base_vel = max(20, base_vel - 6)

            offset_in_score = blk_data.get("offset", 0.0) # Renamed for clarity
            remaining_ql_in_block = blk_data.get("q_length", pattern_unit_length_ql)

            if blk_data.get("is_first_in_section", False) and blk_idx > 0 :
                ms_since_fill = 0
            
            current_pos_within_block = 0.0
            while remaining_ql_in_block > MIN_NOTE_DURATION_QL / 8.0:
                # Duration for this iteration of the pattern unit
                current_pattern_iteration_ql = min(pattern_unit_length_ql, remaining_ql_in_block)
                if current_pattern_iteration_ql < MIN_NOTE_DURATION_QL / 4.0 : break

                # is_last_bar_of_block logic needs to be based on remaining_ql_in_block relative to pattern_unit_length_ql
                is_last_pattern_iteration_in_block = (remaining_ql_in_block <= pattern_unit_length_ql + (MIN_NOTE_DURATION_QL / 8.0))

                pattern_to_use = pat_events
                fill_applied_this_iter = False
                
                override_fill_key = drums_params.get("fill_override", drums_params.get("drum_fill_key_override"))
                if is_last_pattern_iteration_in_block and override_fill_key:
                    chosen_fill_pattern = fills.get(override_fill_key)
                    if chosen_fill_pattern is not None:
                        pattern_to_use = chosen_fill_pattern
                        fill_applied_this_iter = True
                        logger.debug(f"DrumGen _render: Applied override fill '{override_fill_key}' for style '{style_key}'")
                    else: logger.warning(f"DrumGen _render: Override fill key '{override_fill_key}' not in fills for '{style_key}'.")
                
                fill_interval = drums_params.get("fill_interval_bars", 0)
                if not fill_applied_this_iter and is_last_pattern_iteration_in_block and fill_interval > 0:
                    # Assuming fill_interval is in terms of main pattern bars
                    num_pattern_units_for_interval = fill_interval # If pattern_unit_length_ql is one bar
                    # If pattern_unit_length_ql is e.g. 2 bars, and fill_interval is 4 bars, then interval is every 2 pattern units.
                    # This fill logic assumes ms_since_fill counts full applications of pattern_unit_length_ql
                    if (ms_since_fill + 1) >= num_pattern_units_for_interval :
                        fill_keys_list = drums_params.get("fill_keys", [])
                        possible_fills = [fk for fk in fill_keys_list if fk in fills]
                        if possible_fills:
                            chosen_fill_key = self.rng.choice(possible_fills)
                            chosen_fill_pattern = fills.get(chosen_fill_key)
                            if chosen_fill_pattern is not None:
                                pattern_to_use = chosen_fill_pattern
                                fill_applied_this_iter = True
                                logger.debug(f"DrumGen _render: Applied scheduled fill '{chosen_fill_key}' for style '{style_key}'")
                
                self._apply_pattern(part, pattern_to_use, offset_in_score + current_pos_within_block, 
                                    current_pattern_iteration_ql, base_vel,
                                    swing_type, swing_ratio_val, pat_ts if pat_ts else self.global_ts)

                if fill_applied_this_iter: ms_since_fill = 0
                else: ms_since_fill += 1 # Increment for each full pattern unit applied without a fill
                
                current_pos_within_block += current_pattern_iteration_ql
                remaining_ql_in_block -= current_pattern_iteration_ql
    
    def _apply_pattern(self, part:stream.Part, events:List[Dict[str,Any]],
                       bar_start_abs:float, current_bar_len_ql:float, base_vel:int,
                       swing_type:str, swing_ratio:float, pattern_ts:meter.TimeSignature):
        beat_len_ql = pattern_ts.beatDuration.quarterLength if pattern_ts else 1.0

        for ev_def in events:
            if self.rng.random() > ev_def.get("probability", 1.0): continue
            inst_name = ev_def.get("instrument")
            if not inst_name: continue
            rel_offset_in_pattern = float(ev_def.get("offset", 0.0))
            
            if swing_ratio != 0.5:
                # Pass swing_type to _swing method
                rel_offset_in_pattern = self._swing(rel_offset_in_pattern, swing_ratio, beat_len_ql, swing_type)
            
            if rel_offset_in_pattern >= current_bar_len_ql - (MIN_NOTE_DURATION_QL / 16.0): continue
            
            hit_duration_ql = float(ev_def.get("duration", 0.125))
            clipped_duration_ql = min(hit_duration_ql, current_bar_len_ql - rel_offset_in_pattern)
            if clipped_duration_ql < MIN_NOTE_DURATION_QL / 8.0: continue

            vel_val = ev_def.get("velocity")
            vel_factor = ev_def.get("velocity_factor", 1.0)
            final_vel: int
            if vel_val is not None: final_vel = int(vel_val)
            else: final_vel = int(base_vel * float(vel_factor))
            final_vel = max(1, min(127, final_vel))

            drum_hit = self._make_hit(inst_name, final_vel, clipped_duration_ql)
            if not drum_hit: continue

            humanize_setting = ev_def.get("humanize")
            time_delta_from_humanizer = 0.0
            if humanize_setting is not None:
                template_name_for_hit = "drum_tight" # Default if humanize: true
                if isinstance(humanize_setting, str): template_name_for_hit = humanize_setting
                
                if isinstance(humanize_setting, bool) and not humanize_setting: # humanize: false
                    pass # Do not humanize
                else: # humanize: true or humanize: "template_name"
                    original_hit_offset_before_humanize = drum_hit.offset # Should be 0.0
                    drum_hit = apply_humanization_to_element(drum_hit, template_name=template_name_for_hit)
                    time_delta_from_humanizer = drum_hit.offset - original_hit_offset_before_humanize
            
            final_insert_offset = bar_start_abs + rel_offset_in_pattern + time_delta_from_humanizer
            drum_hit.offset = 0.0
            part.insert(final_insert_offset, drum_hit)

    def _swing(self, rel_offset:float, swing_ratio:float, beat_len_ql:float, swing_type:str = "eighth")->float:
        if abs(swing_ratio - 0.5) < 1e-3 or beat_len_ql <= 0:
            return rel_offset
        
        subdivision_duration_ql: float
        if swing_type == "eighth":
            subdivision_duration_ql = beat_len_ql / 2.0
        elif swing_type == "sixteenth":
            subdivision_duration_ql = beat_len_ql / 4.0
        else: # Unsupported swing type, return original offset
            logger.warning(f"DrumGen _swing: Unsupported swing_type '{swing_type}'. No swing applied.")
            return rel_offset

        # Effective beat length for this subdivision pair (e.g., for 8th swing, this is a quarter note; for 16th swing, this is an 8th note)
        effective_beat_for_swing_pair_ql = subdivision_duration_ql * 2.0

        beat_num_in_bar = math.floor(rel_offset / effective_beat_for_swing_pair_ql)
        offset_within_effective_beat = rel_offset - (beat_num_in_bar * effective_beat_for_swing_pair_ql)
        
        epsilon = subdivision_duration_ql * 0.1

        # Check if it's the *second* subdivision of the pair (the one that gets delayed)
        if abs(offset_within_effective_beat - subdivision_duration_ql) < epsilon:
            # Straight position of this off-beat subdivision is `subdivision_duration_ql`
            # Swung position is `effective_beat_for_swing_pair_ql * swing_ratio`
            new_offset_within_effective_beat = effective_beat_for_swing_pair_ql * swing_ratio
            swung_rel_offset = (beat_num_in_bar * effective_beat_for_swing_pair_ql) + new_offset_within_effective_beat
            return swung_rel_offset
        return rel_offset

    def _make_hit(self, name:str, vel:int, ql:float)->Optional[note.Note]:
        mapped_name = name.lower().replace(" ","_").replace("-","_")
        actual_name_for_midi = GHOST_ALIAS.get(mapped_name, mapped_name)
        midi = GM_DRUM_MAP.get(actual_name_for_midi)
        if midi is None:
            logger.warning(f"DrumGen _make_hit: Unknown drum sound '{name}' (mapped to '{actual_name_for_midi}'). MIDI mapping not found.")
            return None
        n = note.Note()
        n.pitch = pitch.Pitch(midi=midi)
        n.duration = m21dur.Duration(quarterLength=max(MIN_NOTE_DURATION_QL / 8.0, ql))
        n.volume = m21volume.Volume(velocity=max(1,min(127,vel)))
        n.offset = 0.0
        return n

# --- END OF FILE ---
