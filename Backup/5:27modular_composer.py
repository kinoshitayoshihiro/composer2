# --- START OF FILE modular_composer.py (ボーカル情報集約機能追加版) ---
import music21
import sys
import os
import json
import argparse
import logging

# music21 のサブモジュールを正しい形式でインポート
import music21.stream as stream
import music21.tempo as tempo
import music21.instrument as m21instrument
import music21.midi as midi
import music21.meter as meter
import music21.key as key
import music21.harmony # harmony.ChordSymbol と harmony.HarmonyException のためにインポート
import music21.pitch # pitch.Pitch のためにインポート
from music21 import exceptions21

from pathlib import Path
from typing import List, Dict, Optional, Any, cast, Sequence
import random

# --- ユーティリティとジェネレータクラスのインポート ---
try:
    from utilities.core_music_utils import get_time_signature_object, sanitize_chord_label
    from generator import (
        PianoGenerator, DrumGenerator, GuitarGenerator, ChordVoicer,
        MelodyGenerator, BassGenerator, VocalGenerator # VocalGenerator もインポートしておく
    )
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import modules: {e}")
    sys.exit(1)
except Exception as e_imp:
    print(f"An unexpected error occurred during module import: {e_imp}")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - [%(levelname)s] - %(module)s.%(funcName)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("modular_composer")

DEFAULT_CONFIG = {
    "global_tempo": 100, "global_time_signature": "4/4", "global_key_tonic": "C", "global_key_mode": "major",
    "parts_to_generate": {
        "piano": True, "drums": True, "guitar": True, "bass": True,
        "chords": True, "melody": False, "vocal": True # vocalパート生成自体は残す
    },
    "default_part_parameters": {
        "piano": {
            "instrument": "Piano",
            "emotion_to_rh_style_keyword": {"default": "simple_block_rh", "quiet_pain_and_nascent_strength": "piano_reflective_arpeggio_rh", "deep_regret_gratitude_and_realization": "piano_chordal_moving_rh", "acceptance_of_love_and_pain_hopeful_belief": "piano_powerful_block_8ths_rh", "self_reproach_regret_deep_sadness": "piano_reflective_arpeggio_rh", "supported_light_longing_for_rebirth": "piano_chordal_moving_rh", "reflective_transition_instrumental_passage": "piano_reflective_arpeggio_rh", "trial_cry_prayer_unbreakable_heart": "piano_powerful_block_8ths_rh", "memory_unresolved_feelings_silence": "piano_reflective_arpeggio_rh", "wavering_heart_gratitude_chosen_strength": "piano_chordal_moving_rh", "reaffirmed_strength_of_love_positive_determination": "piano_powerful_block_8ths_rh", "hope_dawn_light_gentle_guidance": "piano_reflective_arpeggio_rh", "nature_memory_floating_sensation_forgiveness": "piano_reflective_arpeggio_rh", "future_cooperation_our_path_final_resolve_and_liberation": "piano_powerful_block_8ths_rh"},
            "emotion_to_lh_style_keyword": {"default": "simple_root_lh", "quiet_pain_and_nascent_strength": "piano_sustained_root_lh", "deep_regret_gratitude_and_realization": "piano_walking_bass_like_lh", "acceptance_of_love_and_pain_hopeful_belief": "piano_active_octave_bass_lh", "self_reproach_regret_deep_sadness": "piano_sustained_root_lh", "supported_light_longing_for_rebirth": "piano_walking_bass_like_lh", "reflective_transition_instrumental_passage": "piano_sustained_root_lh", "trial_cry_prayer_unbreakable_heart": "piano_active_octave_bass_lh", "memory_unresolved_feelings_silence": "piano_sustained_root_lh", "wavering_heart_gratitude_chosen_strength": "piano_walking_bass_like_lh", "reaffirmed_strength_of_love_positive_determination": "piano_active_octave_bass_lh", "hope_dawn_light_gentle_guidance": "piano_sustained_root_lh", "nature_memory_floating_sensation_forgiveness": "piano_sustained_root_lh", "future_cooperation_our_path_final_resolve_and_liberation": "piano_active_octave_bass_lh"},
            "style_keyword_to_rhythm_key": {"piano_reflective_arpeggio_rh": "piano_flowing_arpeggio_eighths_rh", "piano_chordal_moving_rh": "piano_chordal_moving_rh_pattern", "piano_powerful_block_8ths_rh": "piano_powerful_block_8ths_rh", "simple_block_rh": "piano_block_quarters_simple", "piano_sustained_root_lh": "piano_sustained_root_lh", "piano_walking_bass_like_lh": "piano_walking_bass_like_lh", "piano_active_octave_bass_lh": "piano_active_octave_bass_lh", "simple_root_lh": "piano_lh_quarter_roots", "default_piano_rh_fallback_rhythm": "default_piano_quarters", "default_piano_lh_fallback_rhythm": "piano_lh_whole_notes"},
            "intensity_to_velocity_ranges": {"low": [50,60,55,65], "medium_low": [55,65,60,70], "medium": [60,70,65,75], "medium_high": [65,80,70,85], "high": [70,85,75,90], "high_to_very_high_then_fade": [75,95,80,100], "default": [60,70,65,75]},
            "default_apply_pedal": True, "default_arp_note_ql": 0.5, "default_rh_voicing_style": "closed", "default_lh_voicing_style": "closed",
            "default_rh_target_octave": 4,
            "default_lh_target_octave": 2,
            "default_rh_num_voices": 3, "default_lh_num_voices": 1,
            "default_humanize": True, "default_humanize_rh": True, "default_humanize_lh": True,
            "default_humanize_style_template": "piano_gentle_arpeggio",
            "default_humanize_time_var": 0.01, "default_humanize_dur_perc": 0.02, "default_humanize_vel_var": 4,
            "default_humanize_fbm_time": False, "default_humanize_fbm_scale": 0.005, "default_humanize_fbm_hurst": 0.7
        },
        "drums": {
            "instrument": "Percussion",
            "emotion_to_style_key": {"default_style": "default_drum_pattern", "quiet_pain_and_nascent_strength": "no_drums", "deep_regret_gratitude_and_realization": "ballad_soft_kick_snare_8th_hat", "acceptance_of_love_and_pain_hopeful_belief": "anthem_rock_chorus_16th_hat", "self_reproach_regret_deep_sadness": "no_drums_or_sparse_cymbal", "supported_light_longing_for_rebirth": "rock_ballad_build_up_8th_hat", "reflective_transition_instrumental_passage": "no_drums_or_gentle_cymbal_swell", "trial_cry_prayer_unbreakable_heart": "rock_ballad_build_up_8th_hat", "memory_unresolved_feelings_silence": "no_drums", "wavering_heart_gratitude_chosen_strength": "ballad_soft_kick_snare_8th_hat", "reaffirmed_strength_of_love_positive_determination": "anthem_rock_chorus_16th_hat", "hope_dawn_light_gentle_guidance": "no_drums_or_gentle_cymbal_swell", "nature_memory_floating_sensation_forgiveness": "no_drums_or_sparse_chimes", "future_cooperation_our_path_final_resolve_and_liberation": "anthem_rock_chorus_16th_hat"},
            "intensity_to_base_velocity": {"default": [70,80], "low": [55,65], "medium_low": [60,70], "medium": [70,80], "medium_high": [75,85], "high": [85,95], "high_to_very_high_then_fade": [90,105]},
            "default_fill_interval_bars": 4, "default_fill_keys": ["simple_snare_roll_half_bar", "chorus_end_fill", "expressive_fill", "soulful_tom_roll"],
            "default_humanize": True, "default_humanize_style_template": "drum_loose_fbm",
            "default_humanize_time_var": 0.015, "default_humanize_dur_perc": 0.03, "default_humanize_vel_var": 6,
            "default_humanize_fbm_time": True, "default_humanize_fbm_scale": 0.01, "default_humanize_fbm_hurst": 0.6
        },
        "guitar": {
            "instrument": "Acoustic Guitar",
            "emotion_mode_to_style_map": {"default_default": {"style": "strum_basic", "voicing_style": "standard", "rhythm_key": "guitar_default_quarters"}, "ionian_希望": {"style": "strum_basic", "voicing_style": "open", "rhythm_key": "guitar_folk_strum_simple"}, "dorian_悲しみ": {"style": "arpeggio", "voicing_style": "standard", "arpeggio_type": "updown", "arpeggio_note_duration_ql": 0.5, "rhythm_key": "guitar_ballad_arpeggio"}, "aeolian_怒り": {"style": "muted_rhythm", "voicing_style": "power_chord_root_fifth", "rhythm_key": "guitar_rock_mute_16th"}},
            "default_style": "strum_basic", "default_rhythm_category": "guitar_patterns", "default_rhythm_key": "guitar_default_quarters", "default_voicing_style": "standard", "default_num_strings": 6, "default_target_octave": 3, "default_velocity": 70, "default_arpeggio_type": "up", "default_arpeggio_note_duration_ql": 0.5, "default_strum_delay_ql": 0.02, "default_mute_note_duration_ql": 0.1, "default_mute_interval_ql": 0.25,
            "default_humanize": True, "default_humanize_style_template": "default_guitar_subtle",
            "default_humanize_time_var": 0.015, "default_humanize_dur_perc": 0.04, "default_humanize_vel_var": 6,
            "default_humanize_fbm_time": False, "default_humanize_fbm_scale": 0.01, "default_humanize_fbm_hurst": 0.7
        },
        "vocal": {
            "instrument": "Voice",
            "data_paths": {"midivocal_data_path": "data/vocal_note_data_ore.json"},
            "default_humanize_opt": True, "default_humanize_template_name": "vocal_ballad_smooth",
            "default_humanize_time_var": 0.02, "default_humanize_dur_perc": 0.04, "default_humanize_vel_var": 5,
            "default_humanize_fbm_time": True, "default_humanize_fbm_scale": 0.01, "default_humanize_fbm_hurst": 0.65
        },
        "bass": {
            "instrument": "Acoustic Bass",
            "default_style": "simple_roots", "default_rhythm_key": "bass_quarter_notes", # これは rhythm_library.json の "bass_lines" を参照
            "default_octave": 2, "default_velocity": 70,
            "default_humanize": True, "default_humanize_style_template": "default_subtle",
            "default_humanize_time_var": 0.01, "default_humanize_dur_perc": 0.03, "default_humanize_vel_var": 5
        },
        "melody": {
            "instrument": "Piano",
            "default_rhythm_key": "default_melody_rhythm", # これは rhythm_library.json の "melody_rhythms" を参照
            "default_octave_range": [4,5],
            "default_density": 0.7,
            "default_velocity": 75,
            "default_humanize": True,
            "default_humanize_style_template": "default_subtle",
            "default_humanize_time_var": 0.01,
            "default_humanize_dur_perc": 0.02,
            "default_humanize_vel_var": 4
        },
        "chords": {
            "instrument": "Violin",
            "chord_voicing_style": "closed",
            "chord_target_octave": 3,
            "chord_num_voices": 4,
            "chord_velocity": 64
        }
    },
    "output_filename_template": "output_{song_title}.mid"
}

def load_json_file(file_path: Path, description: str) -> Optional[Dict | List]:
    if not file_path.exists():
        logger.error(f"{description} not found: {file_path}")
        sys.exit(1)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded {description} from: {file_path}")
        return data
    except json.JSONDecodeError as e_json:
        logger.error(f"Error decoding JSON from {description} at {file_path}: {e_json}", exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading {description} from {file_path}: {e}", exc_info=True)
        sys.exit(1)

def _get_humanize_params(params_from_chordmap: Dict[str, Any], default_cfg_instrument: Dict[str, Any], instrument_prefix: str) -> Dict[str, Any]:
    humanize_final_params = {}
    humanize_flag = params_from_chordmap.get(f"{instrument_prefix}_humanize", params_from_chordmap.get("humanize", default_cfg_instrument.get(f"default_humanize", False)))
    humanize_final_params["humanize_opt"] = bool(humanize_flag)

    if humanize_final_params["humanize_opt"]:
        humanize_final_params["template_name"] = params_from_chordmap.get(f"{instrument_prefix}_humanize_style_template", default_cfg_instrument.get("default_humanize_style_template"))

        individual_h_keys = ["time_var", "dur_perc", "vel_var", "fbm_time", "fbm_scale", "fbm_hurst"]
        custom_overrides = {}
        for h_key_suffix in individual_h_keys:
            val_from_map = params_from_chordmap.get(f"{instrument_prefix}_humanize_{h_key_suffix}", params_from_chordmap.get(f"humanize_{h_key_suffix}"))
            if val_from_map is not None:
                custom_overrides[h_key_suffix] = val_from_map
            else:
                custom_overrides[h_key_suffix] = default_cfg_instrument.get(f"default_humanize_{h_key_suffix}")
        humanize_final_params["custom_params"] = custom_overrides
    return humanize_final_params


def translate_keywords_to_params(
        musical_intent: Dict[str, Any], chord_block_specific_hints: Dict[str, Any],
        default_instrument_params: Dict[str, Any], instrument_name_key: str,
        rhythm_library_all_categories: Dict # この引数は現状直接使われていないが、将来的に参照する可能性は残す
) -> Dict[str, Any]:
    params: Dict[str, Any] = default_instrument_params.copy()
    emotion_key = musical_intent.get("emotion", "default").lower()
    intensity_key = musical_intent.get("intensity", "default").lower()
    mode_of_block = chord_block_specific_hints.get("mode_of_block", "major").lower()

    section_instrument_settings = chord_block_specific_hints.get("part_settings", {}).get(instrument_name_key, {})
    params.update(section_instrument_settings)

    humanize_resolved_params = _get_humanize_params(params, default_instrument_params, instrument_name_key)
    params.update(humanize_resolved_params)

    logger.debug(f"Translating for {instrument_name_key}: Emo='{emotion_key}', Int='{intensity_key}', Mode='{mode_of_block}', InitialParams='{params}'")

    if instrument_name_key == "piano":
        cfg_piano = DEFAULT_CONFIG["default_part_parameters"]["piano"]
        if "piano_rh_style_keyword" not in params:
            params["piano_rh_style_keyword"] = cfg_piano.get("emotion_to_rh_style_keyword", {}).get(emotion_key, cfg_piano.get("emotion_to_rh_style_keyword", {}).get("default"))
        if "piano_lh_style_keyword" not in params:
            params["piano_lh_style_keyword"] = cfg_piano.get("emotion_to_lh_style_keyword", {}).get(emotion_key, cfg_piano.get("emotion_to_lh_style_keyword", {}).get("default"))

        rh_style_keyword = params.get("piano_rh_style_keyword")
        lh_style_keyword = params.get("piano_lh_style_keyword")
        if rh_style_keyword and "piano_rh_rhythm_key" not in params :
            params["piano_rh_rhythm_key"] = cfg_piano.get("style_keyword_to_rhythm_key",{}).get(rh_style_keyword, cfg_piano.get("style_keyword_to_rhythm_key",{}).get("default_piano_rh_fallback_rhythm"))
        if lh_style_keyword and "piano_lh_rhythm_key" not in params :
            params["piano_lh_rhythm_key"] = cfg_piano.get("style_keyword_to_rhythm_key",{}).get(lh_style_keyword, cfg_piano.get("style_keyword_to_rhythm_key",{}).get("default_piano_lh_fallback_rhythm"))

        vel_ranges = cfg_piano.get("intensity_to_velocity_ranges", {}).get(intensity_key, cfg_piano.get("intensity_to_velocity_ranges", {}).get("default", [60,70,65,75]))
        if "piano_velocity_rh_min" not in params and len(vel_ranges) > 0: params["piano_velocity_rh_min"] = vel_ranges[0]
        if "piano_velocity_rh_max" not in params and len(vel_ranges) > 1: params["piano_velocity_rh_max"] = vel_ranges[1]
        if "piano_velocity_lh_min" not in params and len(vel_ranges) > 2: params["piano_velocity_lh_min"] = vel_ranges[2]
        if "piano_velocity_lh_max" not in params and len(vel_ranges) > 3: params["piano_velocity_lh_max"] = vel_ranges[3]

        for suffix in ["apply_pedal", "arp_note_ql", "rh_voicing_style", "lh_voicing_style", "rh_target_octave", "lh_target_octave", "rh_num_voices", "lh_num_voices"]:
            param_name = f"piano_{suffix}"
            if param_name not in params: params[param_name] = cfg_piano.get(f"default_{suffix}")

        params["humanize_rh_opt"] = params.get("piano_humanize_rh", params.get("humanize_opt", False))
        params["humanize_lh_opt"] = params.get("piano_humanize_lh", params.get("humanize_opt", False))

    elif instrument_name_key == "drums":
        cfg_drums = DEFAULT_CONFIG["default_part_parameters"]["drums"]
        if "drum_style_key" not in params:
            params["drum_style_key"] = cfg_drums.get("emotion_to_style_key", {}).get(emotion_key, cfg_drums.get("emotion_to_style_key", {}).get("default_style"))
        drum_vel_setting = cfg_drums.get("intensity_to_base_velocity", {}).get(intensity_key, cfg_drums.get("intensity_to_base_velocity", {}).get("default", [80,90]))
        if "drum_base_velocity" not in params:
             params["drum_base_velocity"] = drum_vel_setting[0] if isinstance(drum_vel_setting, list) and drum_vel_setting else drum_vel_setting
        if "drum_fill_interval_bars" not in params: params["drum_fill_interval_bars"] = cfg_drums.get("default_fill_interval_bars")
        if "drum_fill_keys" not in params: params["drum_fill_keys"] = cfg_drums.get("default_fill_keys")

    elif instrument_name_key == "guitar":
        cfg_guitar = DEFAULT_CONFIG["default_part_parameters"]["guitar"]
        emotion_mode_key = f"{mode_of_block}_{emotion_key}"
        style_map = cfg_guitar.get("emotion_mode_to_style_map", {})
        specific_style_config = style_map.get(emotion_mode_key,
                                        style_map.get(emotion_key,
                                            style_map.get(f"default_{mode_of_block}",
                                                style_map.get("default_default", {}))))
        param_keys_guitar = ["guitar_style", "guitar_rhythm_key", "guitar_voicing_style", "guitar_num_strings", "guitar_target_octave", "guitar_velocity", "arpeggio_type", "arpeggio_note_duration_ql", "strum_delay_ql", "mute_note_duration_ql", "mute_interval_ql"]
        for p_key in param_keys_guitar:
            if p_key not in params:
                specific_key_name = p_key.replace("guitar_", "")
                params[p_key] = specific_style_config.get(specific_key_name, cfg_guitar.get(f"default_{specific_key_name}"))
        if "guitar_rhythm_key" not in params or not params["guitar_rhythm_key"]:
            params["guitar_rhythm_key"] = cfg_guitar.get("default_rhythm_key")

    elif instrument_name_key == "vocal":
        pass

    elif instrument_name_key == "bass":
        cfg_bass = DEFAULT_CONFIG["default_part_parameters"]["bass"]
        if "style" not in params: # "bass_style" ではなく "style" で chordmap と合わせる
            params["style"] = cfg_bass.get("style_map",{}).get(emotion_key, cfg_bass.get("style_map",{}).get("default", cfg_bass.get("default_style")))
        if "rhythm_key" not in params: # "bass_rhythm_key" ではなく "rhythm_key"
            params["rhythm_key"] = cfg_bass.get("rhythm_key_map", {}).get(emotion_key, cfg_bass.get("rhythm_key_map", {}).get("default", cfg_bass.get("default_rhythm_key")))
        if "octave" not in params: params["octave"] = cfg_bass.get("default_octave")
        if "velocity" not in params: params["velocity"] = cfg_bass.get("default_velocity")

    elif instrument_name_key == "melody":
        cfg_melody = DEFAULT_CONFIG["default_part_parameters"]["melody"]
        if "rhythm_key" not in params:
            params["rhythm_key"] = cfg_melody.get("rhythm_key_map", {}).get(emotion_key, cfg_melody.get("rhythm_key_map", {}).get("default", cfg_melody.get("default_rhythm_key")))
        if "octave_range" not in params: params["octave_range"] = cfg_melody.get("default_octave_range")
        if "density" not in params: params["density"] = cfg_melody.get("default_density")
        if "velocity" not in params: params["velocity"] = cfg_melody.get("default_velocity")

    block_instrument_specific_hints = chord_block_specific_hints.get(instrument_name_key, {})
    if isinstance(block_instrument_specific_hints, dict):
        params.update(block_instrument_specific_hints)

    if instrument_name_key == "drums" and "drum_fill" in chord_block_specific_hints: # chord_block_specific_hints から直接参照
        params["drum_fill_key_override"] = chord_block_specific_hints["drum_fill"]

    logger.info(f"Final params for [{instrument_name_key}] (Emo: {emotion_key}, Int: {intensity_key}, Mode: {mode_of_block}) -> {params}")
    return params

def parse_vocal_midi_data_for_context(midivocal_data_list: Optional[List[Dict]]) -> List[Dict]:
    parsed_notes = []
    if not midivocal_data_list:
        logger.info("Composer: No vocal MIDI data provided for context parsing.")
        return parsed_notes
    for item_idx, item in enumerate(midivocal_data_list):
        try:
            offset = float(item.get("offset", item.get("Offset", 0.0)))
            pitch_name = str(item.get("pitch", item.get("Pitch", "")))
            length = float(item.get("length", item.get("Length", 0.0)))
            # velocity = int(item.get("velocity", item.get("Velocity", 70))) # 必要に応じて

            if not pitch_name:
                logger.debug(f"Composer: Vocal note item #{item_idx+1} has empty pitch. Skipping.")
                continue
            try:
                music21.pitch.Pitch(pitch_name) # ピッチ名の妥当性チェック
            except Exception as e_pitch_parse:
                logger.warning(f"Composer: Skipping vocal item #{item_idx+1} due to invalid pitch_name '{pitch_name}': {e_pitch_parse}")
                continue
            if length <= 0:
                logger.debug(f"Composer: Vocal note item #{item_idx+1} with pitch '{pitch_name}' has non-positive length {length}. Skipping.")
                continue

            parsed_notes.append({
                "offset": offset,
                "pitch_str": pitch_name,
                "q_length": length,
                # "velocity": velocity
            })
        except KeyError as ke:
            logger.error(f"Composer: Skipping vocal item #{item_idx+1} due to missing key: {ke} in {item}")
        except ValueError as ve:
            logger.error(f"Composer: Skipping vocal item #{item_idx+1} due to ValueError: {ve} in {item}")
        except Exception as e:
            logger.error(f"Composer: Unexpected error parsing vocal item #{item_idx+1}: {e} in {item}", exc_info=True)

    parsed_notes.sort(key=lambda x: x["offset"])
    logger.info(f"Composer: Parsed {len(parsed_notes)} valid notes from vocal MIDI data for context.")
    return parsed_notes

def prepare_processed_stream(chordmap_data: Dict, main_config: Dict, rhythm_lib_all: Dict,
                             parsed_vocal_track: List[Dict]) -> List[Dict]:
    processed_stream: List[Dict] = []
    current_abs_offset: float = 0.0
    g_settings = chordmap_data.get("global_settings", {})
    ts_str = g_settings.get("time_signature", main_config["global_time_signature"])
    ts_obj = get_time_signature_object(ts_str)
    if ts_obj is None:
        logger.error("Failed to get TimeSignature object. Defaulting to 4/4 time.")
        ts_obj = music21.meter.TimeSignature("4/4") # music21.meter を使用
    beats_per_measure = ts_obj.barDuration.quarterLength

    g_key_t, g_key_m = g_settings.get("key_tonic", main_config["global_key_tonic"]), g_settings.get("key_mode", main_config["global_key_mode"])

    sections_items = chordmap_data.get("sections", {}).items()
    sorted_sections = sorted(sections_items, key=lambda item: item[1].get("order", float('inf')) if isinstance(item[1], dict) else float('inf'))

    for sec_name, sec_info_any in sorted_sections:
        if not isinstance(sec_info_any, dict):
            logger.warning(f"Section '{sec_name}' data is not a dictionary. Skipping.")
            continue
        sec_info: Dict[str, Any] = sec_info_any

        logger.info(f"Preparing section: {sec_name}")
        sec_intent = sec_info.get("musical_intent", {})
        sec_part_settings_for_all_instruments = sec_info.get("part_settings", {})
        sec_t, sec_m = sec_info.get("tonic", g_key_t), sec_info.get("mode", g_key_m)
        sec_len_meas = sec_info.get("length_in_measures")
        chord_prog = sec_info.get("chord_progression", [])
        if not chord_prog:
            logger.warning(f"Section '{sec_name}' has no chord_progression. Skipping.")
            continue

        default_beats_per_chord_block: Optional[float] = None
        if sec_len_meas and len(chord_prog) > 0:
            try:
                default_beats_per_chord_block = (float(sec_len_meas) * beats_per_measure) / len(chord_prog)
            except (ValueError, TypeError):
                logger.warning(f"Could not calculate default_beats_per_chord_block for section {sec_name}.")

        for c_idx, c_def_any in enumerate(chord_prog):
            if not isinstance(c_def_any, dict):
                logger.warning(f"Chord definition at index {c_idx} in section '{sec_name}' is not a dictionary. Skipping.")
                continue
            c_def: Dict[str, Any] = c_def_any

            original_chord_label = c_def.get("label", "C")
            sanitized_chord_label_for_block: Optional[str] = None
            is_valid_chord_label_for_block = False

            if not original_chord_label or original_chord_label.strip().lower() in ["rest", "r", "nc", "n.c.", "silence", "-"]:
                logger.info(f"Section '{sec_name}', Chord {c_idx+1}: Label '{original_chord_label}' is a rest.")
                sanitized_chord_label_for_block = "Rest"
                is_valid_chord_label_for_block = True
            else:
                temp_sanitized_label = sanitize_chord_label(original_chord_label)
                if temp_sanitized_label is None:
                    logger.warning(f"Section '{sec_name}', Chord {c_idx+1}: Label '{original_chord_label}' was sanitized to Rest by sanitize_chord_label. Treating as Rest.")
                    sanitized_chord_label_for_block = "Rest"
                    is_valid_chord_label_for_block = True
                else:
                    try:
                        cs = music21.harmony.ChordSymbol(temp_sanitized_label)
                        if cs.pitches:
                            sanitized_chord_label_for_block = temp_sanitized_label
                            is_valid_chord_label_for_block = True
                            logger.debug(f"Section '{sec_name}', Chord {c_idx+1}: Label '{original_chord_label}' (sanitized: '{sanitized_chord_label_for_block}') parsed successfully by music21 -> {cs.figure}")
                        else:
                            logger.warning(f"Section '{sec_name}', Chord {c_idx+1}: Label '{original_chord_label}' (sanitized: '{temp_sanitized_label}') parsed by music21 but resulted in NO PITCHES (figure: {cs.figure}). Treating as invalid.")
                    except music21.harmony.HarmonyException as e_harm:
                        logger.warning(f"Section '{sec_name}', Chord {c_idx+1}: Label '{original_chord_label}' (sanitized: '{temp_sanitized_label}') FAILED music21 parsing. Error: {e_harm}. Treating as invalid.")
                    except Exception as e_other_parse:
                        logger.error(f"Section '{sec_name}', Chord {c_idx+1}: UNEXPECTED error parsing label '{original_chord_label}' (sanitized: '{temp_sanitized_label}'): {e_other_parse}. Treating as invalid.", exc_info=True)

            if not is_valid_chord_label_for_block:
                logger.error(f"Section '{sec_name}', Chord {c_idx+1}: Due to parsing issues, invalid chord label '{original_chord_label}' will be treated as 'C'. Please review chordmap.json.")
                c_lbl = "C"
            else:
                c_lbl = cast(str, sanitized_chord_label_for_block)

            dur_b_val = c_def.get("duration_beats")
            if dur_b_val is not None:
                try: dur_b = float(dur_b_val)
                except (ValueError, TypeError):
                    logger.warning(f"Invalid duration_beats '{dur_b_val}' for chord '{original_chord_label}' in section '{sec_name}'. Using default.")
                    dur_b = default_beats_per_chord_block if default_beats_per_chord_block is not None else beats_per_measure
            elif default_beats_per_chord_block is not None: dur_b = default_beats_per_chord_block
            else: dur_b = beats_per_measure

            blk_intent = sec_intent.copy();
            if "emotion" in c_def: blk_intent["emotion"] = c_def["emotion"]
            if "intensity" in c_def: blk_intent["intensity"] = c_def["intensity"]
            blk_hints_for_translate = {"part_settings": sec_part_settings_for_all_instruments.copy()}
            current_block_mode = c_def.get("mode", sec_m)
            blk_hints_for_translate["mode_of_block"] = current_block_mode
            reserved_keys = {"label", "duration_beats", "order", "musical_intent", "part_settings", "tensions_to_add", "emotion", "intensity", "mode"}
            for k_hint, v_hint in c_def.items():
                if k_hint not in reserved_keys: blk_hints_for_translate[k_hint] = v_hint

            vocal_notes_in_this_block = []
            block_start_time = current_abs_offset
            block_end_time = current_abs_offset + dur_b
            for vocal_note in parsed_vocal_track:
                v_offset = vocal_note["offset"]
                v_end_offset = v_offset + vocal_note["q_length"]
                if max(block_start_time, v_offset) < min(block_end_time, v_end_offset):
                    relative_v_offset = v_offset - block_start_time
                    vocal_notes_in_this_block.append({
                        "pitch_str": vocal_note["pitch_str"],
                        "q_length": vocal_note["q_length"],
                        "block_relative_offset": relative_v_offset,
                        "absolute_offset": v_offset
                    })

            blk_data = {
                "offset": current_abs_offset, "q_length": dur_b, "chord_label": c_lbl,
                "section_name": sec_name, "tonic_of_section": sec_t, "mode": current_block_mode,
                "is_first_in_section":(c_idx==0), "is_last_in_section":(c_idx==len(chord_prog)-1),
                "vocal_notes_in_block": vocal_notes_in_this_block,
                "part_params":{}
            }
            for p_key_name, generate_flag in main_config.get("parts_to_generate", {}).items():
                if generate_flag:
                    default_params_for_instrument = main_config["default_part_parameters"].get(p_key_name, {})
                    chord_specific_settings_for_part = c_def.get("part_specific_hints", {}).get(p_key_name, {})
                    final_hints_for_translate = blk_hints_for_translate.copy()
                    final_hints_for_translate.update(chord_specific_settings_for_part)
                    blk_data["part_params"][p_key_name] = translate_keywords_to_params(
                        blk_intent, final_hints_for_translate, default_params_for_instrument,
                        p_key_name, rhythm_lib_all
                    )
            processed_stream.append(blk_data)
            current_abs_offset += dur_b
    logger.info(f"Prepared {len(processed_stream)} blocks. Total duration: {current_abs_offset:.2f} beats.")
    return processed_stream

def run_composition(cli_args: argparse.Namespace, main_cfg: Dict, chordmap_data: Dict, rhythm_lib_data: Dict):
    logger.info("=== Running Main Composition Workflow ===")
    final_score = stream.Score()
    final_score.insert(0, tempo.MetronomeMark(number=main_cfg["global_tempo"]))
    try:
        ts_obj_score = get_time_signature_object(main_cfg["global_time_signature"]);
        if ts_obj_score: final_score.insert(0, ts_obj_score)
        else: final_score.insert(0, meter.TimeSignature("4/4"))
        key_t, key_m = main_cfg["global_key_tonic"], main_cfg["global_key_mode"]
        sections_data = chordmap_data.get("sections")
        if sections_data and isinstance(sections_data, dict) and sections_data:
            try:
                first_sec_name = sorted(sections_data.items(), key=lambda item: item[1].get("order", float('inf')) if isinstance(item[1], dict) else float('inf'))[0][0]
                first_sec_info = sections_data[first_sec_name]
                if isinstance(first_sec_info, dict):
                    key_t = first_sec_info.get("tonic", key_t)
                    key_m = first_sec_info.get("mode", key_m)
            except IndexError: logger.warning("No sections for initial key.")
            except Exception as e_key_init: logger.warning(f"Could not determine initial key: {e_key_init}.")
        final_score.insert(0, key.Key(key_t, key_m.lower()))
    except Exception as e:
        logger.error(f"Error setting score globals: {e}. Defaults.", exc_info=True)
        if not final_score.getElementsByClass(meter.TimeSignature).first(): final_score.insert(0, meter.TimeSignature("4/4"))
        if not final_score.getElementsByClass(key.Key).first(): final_score.insert(0, key.Key(main_cfg.get("global_key_tonic","C"), main_cfg.get("global_key_mode","major")))

    parsed_vocal_track_for_context: List[Dict] = []
    vocal_data_paths = main_cfg.get("default_part_parameters", {}).get("vocal", {}).get("data_paths", {})
    midivocal_p_str_context = cli_args.vocal_mididata_path or chordmap_data.get("global_settings",{}).get("vocal_mididata_path", vocal_data_paths.get("midivocal_data_path"))
    if midivocal_p_str_context:
        vocal_midi_data_content = load_json_file(Path(str(midivocal_p_str_context)), "Vocal MIDI Data for Context")
        if isinstance(vocal_midi_data_content, list):
            parsed_vocal_track_for_context = parse_vocal_midi_data_for_context(vocal_midi_data_content)
        else: logger.warning(f"Vocal MIDI data for context at '{midivocal_p_str_context}' is not a list.")
    if not parsed_vocal_track_for_context: logger.info("No vocal track data for context. Bass/Melody generated without direct vocal reference.")

    proc_blocks = prepare_processed_stream(chordmap_data, main_cfg, rhythm_lib_data, parsed_vocal_track_for_context)

    if not proc_blocks: logger.error("No blocks to process. Aborting."); return
    cv_inst = ChordVoicer(global_tempo=main_cfg["global_tempo"], global_time_signature=main_cfg["global_time_signature"])
    gens: Dict[str, Any] = {}

    for part_name, generate_flag in main_cfg.get("parts_to_generate", {}).items():
        if not generate_flag: continue
        part_default_cfg = main_cfg["default_part_parameters"].get(part_name, {})
        instrument_str = part_default_cfg.get("instrument", "Piano")
        rhythm_category_key: Optional[str] = None; rhythm_lib_for_instrument: Dict[str, Any] = {}
        if part_name == "drums": rhythm_category_key = "drum_patterns"
        elif part_name == "bass": rhythm_category_key = "bass_lines"
        elif part_name == "melody": rhythm_category_key = "melody_rhythms"
        elif part_name == "piano": rhythm_category_key = "piano_patterns"
        elif part_name == "guitar": rhythm_category_key = "guitar_patterns"
        if rhythm_category_key:
            rhythm_lib_for_instrument = rhythm_lib_data.get(rhythm_category_key, {})
            if not rhythm_lib_for_instrument: logger.warning(f"Rhythm category '{rhythm_category_key}' for '{part_name}' not in rhythm_library or empty.")
            else: logger.info(f"Loaded {len(rhythm_lib_for_instrument)} patterns for '{rhythm_category_key}' for '{part_name}'.")
        else: logger.info(f"Part '{part_name}' does not use predefined rhythm library category.")

        instrument_obj = None
        try: instrument_obj = m21instrument.fromString(instrument_str)
        except exceptions21.InstrumentException:
            logger.warning(f"Could not match '{instrument_str}'. Trying direct class.")
            try:
                instrument_class = getattr(m21instrument, instrument_str.replace(" ", ""), None)
                if instrument_class and callable(instrument_class): instrument_obj = instrument_class()
                else: logger.warning(f"Class '{instrument_str.replace(' ', '')}' not in m21instrument. Default Piano."); instrument_obj = m21instrument.Piano()
            except Exception as e_getattr: logger.error(f"Error instantiating '{instrument_str}': {e_getattr}. Default Piano."); instrument_obj = m21instrument.Piano()
        except Exception as e_other_inst: logger.error(f"Unexpected error with '{instrument_str}': {e_other_inst}. Default Piano."); instrument_obj = m21instrument.Piano()

        if part_name == "piano": gens[part_name] = PianoGenerator(rhythm_library=cast(Dict[str,Dict], rhythm_lib_for_instrument), chord_voicer_instance=cv_inst, default_instrument_rh=instrument_obj, default_instrument_lh=instrument_obj, global_tempo=main_cfg["global_tempo"], global_time_signature=main_cfg["global_time_signature"])
        elif part_name == "drums":
            gens[part_name] = DrumGenerator(
                lib=cast(Dict[str,Dict[str,Any]], rhythm_lib_for_instrument),  # "drum_pattern_library" を "lib" に変更
                tempo_bpm=main_cfg["global_tempo"],                             # "global_tempo" を "tempo_bpm" に変更
                time_sig=main_cfg["global_time_signature"]                    # "global_time_signature" を "time_sig" に変更
                # "default_instrument" は新しい __init__ から削除されたため、ここからも削除
        )
        elif part_name == "guitar": gens[part_name] = GuitarGenerator(rhythm_library=cast(Dict[str,Dict], rhythm_lib_for_instrument), default_instrument=instrument_obj, global_tempo=main_cfg["global_tempo"], global_time_signature=main_cfg["global_time_signature"])
        elif part_name == "vocal":
            if main_cfg["parts_to_generate"].get("vocal"):
                gens[part_name] = VocalGenerator(default_instrument=instrument_obj, global_tempo=main_cfg["global_tempo"], global_time_signature=main_cfg["global_time_signature"])
        elif part_name == "bass": gens[part_name] = BassGenerator(rhythm_library=cast(Dict[str,Dict], rhythm_lib_for_instrument), default_instrument=instrument_obj, global_tempo=main_cfg["global_tempo"], global_time_signature=main_cfg["global_time_signature"], global_key_tonic=main_cfg["global_key_tonic"], global_key_mode=main_cfg["global_key_mode"])
        elif part_name == "melody": gens[part_name] = MelodyGenerator(rhythm_library=cast(Dict[str,Dict], rhythm_lib_for_instrument), default_instrument=instrument_obj, global_tempo=main_cfg["global_tempo"], global_time_signature=main_cfg["global_time_signature"], global_key_signature_tonic=main_cfg["global_key_tonic"], global_key_signature_mode=main_cfg["global_key_mode"])
        elif part_name == "chords": gens[part_name] = cv_inst;
        if part_name == "chords" and instrument_obj : cv_inst.default_instrument = instrument_obj

    for p_n, p_g_inst in gens.items():
        if p_g_inst and main_cfg["parts_to_generate"].get(p_n):
            logger.info(f"Generating {p_n} part...")
            try:
                part_obj: Optional[stream.Stream] = None
                if p_n == "vocal":
                    vocal_params_for_compose = proc_blocks[0]["part_params"].get("vocal") if proc_blocks else main_cfg["default_part_parameters"].get("vocal", {})
                    midivocal_data_for_compose_list : Optional[List[Dict]] = None
                    vocal_data_paths_call = main_cfg["default_part_parameters"].get("vocal", {}).get("data_paths", {})
                    midivocal_p_str_call = cli_args.vocal_mididata_path or chordmap_data.get("global_settings",{}).get("vocal_mididata_path", vocal_data_paths_call.get("midivocal_data_path"))
                    if midivocal_p_str_call:
                        loaded_data = load_json_file(Path(str(midivocal_p_str_call)), "Vocal MIDI Data for VocalGenerator.compose")
                        if isinstance(loaded_data, list): midivocal_data_for_compose_list = loaded_data
                    if midivocal_data_for_compose_list:
                        part_obj = p_g_inst.compose(
                            midivocal_data=midivocal_data_for_compose_list,
                            processed_chord_stream=proc_blocks,
                            humanize_opt=vocal_params_for_compose.get("humanize_opt", True),
                            humanize_template_name=vocal_params_for_compose.get("template_name"),
                            humanize_custom_params=vocal_params_for_compose.get("custom_params")
                        )
                    else: logger.warning(f"Vocal generation skipped in compose: No MIDI data from '{midivocal_p_str_call}'."); continue
                else:
                    part_obj = p_g_inst.compose(proc_blocks)

                if isinstance(part_obj, stream.Score) and part_obj.parts:
                    for sub_part in part_obj.parts:
                        if sub_part.flatten().notesAndRests: final_score.insert(0, sub_part)
                elif isinstance(part_obj, stream.Part) and part_obj.flatten().notesAndRests:
                    final_score.insert(0, part_obj)
                logger.info(f"{p_n} part generated.")
            except Exception as e_gen: logger.error(f"Error in {p_n} generation: {e_gen}", exc_info=True)

    title = chordmap_data.get("project_title","untitled").replace(" ","_").lower()
    out_fname_template = main_cfg.get("output_filename_template", "output_{song_title}.mid")
    actual_out_fname = cli_args.output_filename if cli_args.output_filename else out_fname_template.format(song_title=title)
    out_fpath = cli_args.output_dir / actual_out_fname
    out_fpath.parent.mkdir(parents=True,exist_ok=True)
    try:
        if final_score.flatten().notesAndRests:
            final_score.write('midi',fp=str(out_fpath))
            logger.info(f"🎉 MIDI exported to {out_fpath}")
        else: logger.warning(f"Score is empty. No MIDI file generated at {out_fpath}.")
    except exceptions21.Music21Exception as e_m21write: logger.error(f"Music21 MIDI write error to {out_fpath}: {e_m21write}", exc_info=True)
    except Exception as e_w: logger.error(f"General MIDI write error to {out_fpath}: {e_w}", exc_info=True)

def main_cli():
    parser = argparse.ArgumentParser(description="Modular Music Composer")
    parser.add_argument("chordmap_file", type=Path, help="Path to the chordmap JSON file.")
    parser.add_argument("rhythm_library_file", type=Path, help="Path to the rhythm library JSON file.")
    parser.add_argument("--output-dir", type=Path, default=Path("midi_output"), help="Directory to save the output MIDI file.")
    parser.add_argument("--output-filename", type=str, help="Custom filename for the output MIDI file.")
    parser.add_argument("--settings-file", type=Path, help="Path to a custom settings JSON file to override defaults.")
    parser.add_argument("--tempo", type=int, help="Override global tempo defined in chordmap or DEFAULT_CONFIG.")
    parser.add_argument("--vocal-mididata-path", type=str, help="Path to vocal MIDI data JSON (overrides config).")
    parser.add_argument("--vocal-lyrics-path", type=str, help="Path to lyrics list JSON (DEPRECATED - no longer used by VocalGenerator).")

    default_parts_cfg = DEFAULT_CONFIG.get("parts_to_generate", {})
    for part_key, default_enabled_status in default_parts_cfg.items():
        arg_name_for_part = f"generate_{part_key}"
        if default_enabled_status: parser.add_argument(f"--no-{part_key}", action="store_false", dest=arg_name_for_part, help=f"Disable {part_key} generation.")
        else: parser.add_argument(f"--include-{part_key}", action="store_true", dest=arg_name_for_part, help=f"Enable {part_key} generation.")
    arg_defaults = {f"generate_{k}": v for k, v in default_parts_cfg.items()}
    parser.set_defaults(**arg_defaults)
    args = parser.parse_args()
    effective_cfg = json.loads(json.dumps(DEFAULT_CONFIG))

    if args.settings_file and args.settings_file.exists():
        custom_settings_data = load_json_file(args.settings_file, "Custom settings")
        if custom_settings_data and isinstance(custom_settings_data, dict):
            def _deep_update(target_dict, source_dict):
                for key_item, value_item in source_dict.items():
                    if isinstance(value_item, dict) and key_item in target_dict and isinstance(target_dict[key_item], dict): _deep_update(target_dict[key_item], value_item)
                    else: target_dict[key_item] = value_item
            _deep_update(effective_cfg, custom_settings_data)

    for pk_name in default_parts_cfg.keys():
        arg_name_cli = f"generate_{pk_name}"
        if hasattr(args, arg_name_cli) and getattr(args, arg_name_cli) is not None:
             effective_cfg["parts_to_generate"][pk_name] = getattr(args, arg_name_cli)

    if args.vocal_mididata_path:
        if "vocal" in effective_cfg["default_part_parameters"] and "data_paths" in effective_cfg["default_part_parameters"]["vocal"]:
            effective_cfg["default_part_parameters"]["vocal"]["data_paths"]["midivocal_data_path"] = str(args.vocal_mididata_path)
    if args.vocal_lyrics_path: logger.warning("Cmd arg --vocal-lyrics-path is deprecated.")

    chordmap_data_loaded = load_json_file(args.chordmap_file, "Chordmap")
    rhythm_library_data_loaded = load_json_file(args.rhythm_library_file, "Rhythm Library")
    if not chordmap_data_loaded or not rhythm_library_data_loaded: logger.critical("Data files missing. Exit."); sys.exit(1)
    if not isinstance(chordmap_data_loaded, dict): logger.critical("Chordmap not dict. Exit."); sys.exit(1)
    if not isinstance(rhythm_library_data_loaded, dict): logger.critical("Rhythm lib not dict. Exit."); sys.exit(1)

    cm_globals_loaded = chordmap_data_loaded.get("global_settings", {})
    effective_cfg["global_tempo"]=cm_globals_loaded.get("tempo",effective_cfg["global_tempo"])
    effective_cfg["global_time_signature"]=cm_globals_loaded.get("time_signature",effective_cfg["global_time_signature"])
    effective_cfg["global_key_tonic"]=cm_globals_loaded.get("key_tonic",effective_cfg["global_key_tonic"])
    effective_cfg["global_key_mode"]=cm_globals_loaded.get("key_mode",effective_cfg["global_key_mode"])
    if args.tempo is not None: effective_cfg["global_tempo"] = args.tempo
    logger.info(f"Final Effective Config: {json.dumps(effective_cfg, indent=2, ensure_ascii=False)}")

    try: run_composition(args, effective_cfg, chordmap_data_loaded, rhythm_library_data_loaded)
    except SystemExit: raise
    except Exception as e_main_run: logger.critical(f"Critical error in main run: {e_main_run}", exc_info=True); sys.exit(1)

if __name__ == "__main__":
    main_cli()
# --- END OF FILE modular_composer.py ---
