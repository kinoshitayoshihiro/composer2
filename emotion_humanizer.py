# --- START OF FILE emotion_humanizer.py (Pydanticモデルとロジック修正版) ---
# -*- coding: utf-8 -*-
import yaml
import random
from pathlib import Path
from typing import Dict, Any, List, Optional, Literal # Literal をインポート
from pydantic import BaseModel, Field, ValidationError
from music21 import harmony, pitch
import re
import logging
import json

# (sanitize_chord_label のインポートとフォールバックは変更なし)
try:
    from utilities.core_music_utils import sanitize_chord_label
except ImportError:
    logging.warning("emotion_humanizer: Could not import sanitize_chord_label. Using basic fallback.")
    def sanitize_chord_label(label: Optional[str]) -> Optional[str]:
        if label is None or not str(label).strip(): return "Rest"
        s = str(label).strip().replace(' ', '')
        if s.upper() in ['NC', 'N.C.', 'NOCHORD', 'SILENCE', '-', 'REST']: return "Rest"
        s = s.replace('Bb', 'B-').replace('Eb', 'E-').replace('Ab', 'A-')
        s = s.replace('Db', 'D-').replace('Gb', 'G-')
        s = s.replace('△', 'maj').replace('M', 'maj')
        if 'majaj' in s: s = s.replace('majaj', 'maj')
        s = s.replace('ø', 'm7b5').replace('Φ', 'm7b5')
        s = s.replace('(', '').replace(')', '')
        return s

logger = logging.getLogger(__name__)

# --- 1. Pydanticモデル定義 ---
class MusicalIntent(BaseModel): # 変更なし
    emotion: str
    intensity: str

class ExpressionDetails(BaseModel): # ★★★ ここを修正 ★★★
    # section_tonic と section_mode はこのモデルからは削除
    # recommended_tensions: List[str] = Field(default_factory=list) # これもマップやロジックで生成するならOptionalでよい
    recommended_tensions: Optional[List[str]] = None # オプショナルに変更
    target_rhythm_category: Optional[str] = None
    approach_style: Optional[str] = None
    articulation_profile: Optional[str] = None
    humanize_profile: Optional[str] = None # 揺らぎの基本プロファイルキー
    
    dynamics_profile: Optional[str] = None
    tempo_feel_adjust_bpm: Optional[float] = None
    # base_articulation の Literal を拡張
    base_articulation: Optional[Literal[
        "legato", "staccato", "tenuto", "accented", "normal", "slurred",
        "accented_legato", "detached_legato" # 新しい値を Literal に追加
    ]] = None
    
    voicing_style_piano_rh: Optional[str] = None
    voicing_style_piano_lh: Optional[str] = None
    voicing_style_guitar: Optional[str] = None

class ChordItem(BaseModel): # 変更なし
    label: str
    duration_beats: float
    nuance: Optional[str] = None

class Section(BaseModel): # 変更なし (前回修正済み)
    order: int
    length_in_measures: int
    tonic: str
    mode: str
    musical_intent: MusicalIntent
    expression_details: Optional[Dict[str, Any]] = None # JSONからは辞書として読み込む想定
    part_settings: Optional[Dict[str, Any]] = Field(default_factory=dict)
    part_specific_hints: Optional[Dict[str, Any]] = Field(default_factory=dict)
    chord_progression: List[ChordItem]
    adjusted_start_beat: Optional[float] = None

class GlobalSettings(BaseModel): # 変更なし
    tempo: int
    time_signature: str
    key_tonic: str
    key_mode: str

class ChordMapInput(BaseModel): # 変更なし
    project_title: Optional[str] = None
    global_settings: GlobalSettings
    sections: Dict[str, Section]

# --- 2. 感情表現プロファイル定義 (既存: 揺らぎ中心) --- (変更なし)
class EmotionExpressionProfile(BaseModel):
    onset_shift_ms: float = 0.0
    sustain_factor: float = 1.0
    velocity_bias: int = 0
    articulation: Optional[Literal["legato", "staccato", "tenuto", "accented", "normal"]] = "normal"

EXISTING_EMOTION_EXPRESSIONS: Dict[str, EmotionExpressionProfile] = {
    "default": EmotionExpressionProfile(),
    "quiet_pain_and_nascent_strength": EmotionExpressionProfile(onset_shift_ms=15, sustain_factor=1.1, velocity_bias=-8, articulation="legato"),
    "deep_regret_gratitude_and_realization": EmotionExpressionProfile(onset_shift_ms=10, sustain_factor=1.0, velocity_bias=2, articulation="tenuto"),
    "acceptance_of_love_and_pain_hopeful_belief": EmotionExpressionProfile(onset_shift_ms=0, sustain_factor=1.0, velocity_bias=5, articulation="legato"),
    "self_reproach_regret_deep_sadness": EmotionExpressionProfile(onset_shift_ms=20, sustain_factor=1.1, velocity_bias=-10, articulation="legato"),
    "supported_light_longing_for_rebirth": EmotionExpressionProfile(onset_shift_ms=-5, sustain_factor=1.1, velocity_bias=4, articulation="legato"),
    "reflective_transition_instrumental_passage": EmotionExpressionProfile(onset_shift_ms=10, sustain_factor=1.05, velocity_bias=-2, articulation="legato"),
    "trial_cry_prayer_unbreakable_heart": EmotionExpressionProfile(onset_shift_ms=-10, sustain_factor=0.9, velocity_bias=8, articulation="accented"),
    "memory_unresolved_feelings_silence": EmotionExpressionProfile(onset_shift_ms=15, sustain_factor=1.15, velocity_bias=-12, articulation="legato"),
    "wavering_heart_gratitude_chosen_strength": EmotionExpressionProfile(onset_shift_ms=0, sustain_factor=1.0, velocity_bias=3, articulation="legato"),
    "reaffirmed_strength_of_love_positive_determination": EmotionExpressionProfile(onset_shift_ms=-10, sustain_factor=0.95, velocity_bias=10, articulation="accented"),
    "hope_dawn_light_gentle_guidance": EmotionExpressionProfile(onset_shift_ms=5, sustain_factor=1.05, velocity_bias=2, articulation="legato"),
    "nature_memory_floating_sensation_forgiveness": EmotionExpressionProfile(onset_shift_ms=10, sustain_factor=1.1, velocity_bias=-5, articulation="legato"),
    "future_cooperation_our_path_final_resolve_and_liberation": EmotionExpressionProfile(onset_shift_ms=0, sustain_factor=1.0, velocity_bias=7, articulation="tenuto"),
    "soft_reflective": EmotionExpressionProfile(onset_shift_ms=18, sustain_factor=1.12, velocity_bias=-7, articulation="legato"),
    "gentle_swing": EmotionExpressionProfile(onset_shift_ms=5, sustain_factor=1.0, velocity_bias=-2, articulation="tenuto"),
    "forward_driving": EmotionExpressionProfile(onset_shift_ms=-2, sustain_factor=0.98, velocity_bias=6, articulation="accented"),
    "delayed_emotion": EmotionExpressionProfile(onset_shift_ms=22, sustain_factor=1.08, velocity_bias=-9, articulation="legato"),
    "gentle_push": EmotionExpressionProfile(onset_shift_ms=3, sustain_factor=1.03, velocity_bias=1, articulation="legato"),
    "bold_drive": EmotionExpressionProfile(onset_shift_ms=-12, sustain_factor=0.92, velocity_bias=12, articulation="accented"),
    "emotional_arc": EmotionExpressionProfile(onset_shift_ms=0, sustain_factor=1.0, velocity_bias=5, articulation="tenuto"),
}


# --- 3. Chordmapチーム提案の感情→演奏解釈マッピング --- (変更なし)
NEW_EMOTION_TO_EXPRESSION_DETAILS_MAP: Dict[str, Dict[str, Any]] = {
    "quiet_pain_and_nascent_strength": {"dynamics_profile": "mp", "base_articulation": "legato", "tempo_feel_adjust_bpm": -2.0, "humanize_profile": "quiet_pain_and_nascent_strength"},
    "deep_regret_gratitude_and_realization": {"dynamics_profile": "p", "base_articulation": "tenuto", "tempo_feel_adjust_bpm": -1.0, "humanize_profile": "deep_regret_gratitude_and_realization"},
    "acceptance_of_love_and_pain_hopeful_belief": {"dynamics_profile": "mf", "base_articulation": "legato", "tempo_feel_adjust_bpm": 0.0, "humanize_profile": "acceptance_of_love_and_pain_hopeful_belief"},
    "self_reproach_regret_deep_sadness": {"dynamics_profile": "pp", "base_articulation": "legato", "tempo_feel_adjust_bpm": -3.0, "humanize_profile": "self_reproach_regret_deep_sadness"},
    "supported_light_longing_for_rebirth": {"dynamics_profile": "mf", "base_articulation": "legato", "tempo_feel_adjust_bpm": 1.0, "humanize_profile": "supported_light_longing_for_rebirth"},
    "reflective_transition_instrumental_passage": {"dynamics_profile": "mp", "base_articulation": "slurred", "tempo_feel_adjust_bpm": 0.0, "humanize_profile": "reflective_transition_instrumental_passage"},
    "trial_cry_prayer_unbreakable_heart": {"dynamics_profile": "f", "base_articulation": "accented", "tempo_feel_adjust_bpm": 2.5, "humanize_profile": "trial_cry_prayer_unbreakable_heart"},
    "memory_unresolved_feelings_silence": {"dynamics_profile": "p", "base_articulation": "legato", "tempo_feel_adjust_bpm": -1.5, "humanize_profile": "memory_unresolved_feelings_silence"},
    "wavering_heart_gratitude_chosen_strength": {"dynamics_profile": "mf", "base_articulation": "legato", "tempo_feel_adjust_bpm": 0.5, "humanize_profile": "wavering_heart_gratitude_chosen_strength"},
    "reaffirmed_strength_of_love_positive_determination": {"dynamics_profile": "ff", "base_articulation": "accented", "tempo_feel_adjust_bpm": 3.0, "humanize_profile": "reaffirmed_strength_of_love_positive_determination"},
    "hope_dawn_light_gentle_guidance": {"dynamics_profile": "mf", "base_articulation": "slurred", "tempo_feel_adjust_bpm": 1.5, "humanize_profile": "hope_dawn_light_gentle_guidance"},
    "nature_memory_floating_sensation_forgiveness": {"dynamics_profile": "mp", "base_articulation": "legato", "tempo_feel_adjust_bpm": -0.5, "humanize_profile": "nature_memory_floating_sensation_forgiveness"},
    "future_cooperation_our_path_final_resolve_and_liberation": {"dynamics_profile": "f-pp", "base_articulation": "tenuto", "tempo_feel_adjust_bpm": -1.0, "humanize_profile": "future_cooperation_our_path_final_resolve_and_liberation"},
    "default_expression_details": {"dynamics_profile": "mf", "base_articulation": "normal", "tempo_feel_adjust_bpm": 0.0, "humanize_profile": "default"}
}

# --- 4. コード解釈と感情表現適用の関数 --- (get_interpreted_chord_details, apply_emotional_expression_to_event は変更なし)
def get_bpm_from_chordmap(chordmap: ChordMapInput) -> float:
    return float(chordmap.global_settings.tempo)

def get_interpreted_chord_details(
    original_chord_label: str,
    recommended_tensions: Optional[List[str]] # Optionalに変更
) -> Dict[str, Optional[str]]:
    if not original_chord_label or original_chord_label.strip().lower() in ["rest", "n.c.", "nc", "none", "-"]:
        return {"interpreted_symbol": "Rest", "specified_bass": None}
    sanitized_label = sanitize_chord_label(original_chord_label)
    if not sanitized_label or sanitized_label == "Rest":
        return {"interpreted_symbol": "Rest", "specified_bass": None}
    current_label = sanitized_label
    base_chord_part = current_label
    bass_note_part = None
    if '/' in current_label:
        parts = current_label.split('/', 1)
        base_chord_part = parts[0]
        if len(parts) > 1:
            bass_note_part = parts[1]
            sanitized_bass = sanitize_chord_label(bass_note_part)
            bass_note_part = sanitized_bass if sanitized_bass != "Rest" else None
    final_symbol_str = base_chord_part
    try:
        cs_test = harmony.ChordSymbol(final_symbol_str)
        if bass_note_part: cs_test.bass(bass_note_part)
    except Exception as e:
        logger.warning(f"  Could not fully validate interpreted symbol '{final_symbol_str}' with bass '{bass_note_part}' using music21: {e}")
    return {"interpreted_symbol": final_symbol_str, "specified_bass": bass_note_part}

def apply_emotional_expression_to_event(
    base_duration_beats: float,
    base_offset_beats: float,
    humanize_profile_key: str,
    section_base_articulation: Optional[str],
    bpm: float,
    base_velocity: int = 64
) -> Dict[str, Any]:
    profile_for_humanize = EXISTING_EMOTION_EXPRESSIONS.get(humanize_profile_key, EXISTING_EMOTION_EXPRESSIONS["default"])
    onset_shift_beats = (profile_for_humanize.onset_shift_ms * bpm) / 60000.0
    actual_offset_beats = base_offset_beats + onset_shift_beats
    actual_duration_beats = base_duration_beats * profile_for_humanize.sustain_factor
    actual_velocity = min(127, max(1, base_velocity + profile_for_humanize.velocity_bias))
    final_articulation = profile_for_humanize.articulation
    if final_articulation == "normal" or final_articulation is None:
        final_articulation = section_base_articulation if section_base_articulation else "normal"
    return {
        "original_duration_beats": round(base_duration_beats, 4),
        "original_offset_beats": round(base_offset_beats, 4),
        "humanized_duration_beats": round(actual_duration_beats, 4),
        "humanized_offset_beats": round(actual_offset_beats, 4),
        "humanized_velocity": actual_velocity,
        "humanized_articulation": final_articulation,
        "emotion_profile_applied": profile_for_humanize.model_dump()
    }

# --- 5. メイン処理関数 ---
def process_chordmap_for_emotion(input_json_path: str, output_yaml_path: str):
    try:
        with Path(input_json_path).open("r", encoding="utf-8") as f:
            raw_chordmap = json.load(f)
        chordmap_model = ChordMapInput.model_validate(raw_chordmap)
    except ValidationError as e_val:
        print(f"Pydantic validation error loading chordmap from {input_json_path}:")
        print(e_val) # 詳細なバリデーションエラー情報を表示
        return
    except Exception as e:
        print(f"Error loading or validating chordmap from {input_json_path}: {e}")
        return

    bpm = get_bpm_from_chordmap(chordmap_model)
    output_data = {
        "project_title": chordmap_model.project_title,
        "global_settings": chordmap_model.global_settings.model_dump(),
        "sections": {}
    }
    current_absolute_offset_beats = 0.0
    sorted_sections = sorted(chordmap_model.sections.items(), key=lambda item: item[1].order)

    for section_name, section_data in sorted_sections: # section_data は Section モデルのインスタンス
        logger.info(f"Processing section: {section_name} (Emotion: {section_data.musical_intent.emotion}, Intensity: {section_data.musical_intent.intensity})")

        # ▼▼▼ expression_details の生成ロジックを修正 ▼▼▼
        # section_data.expression_details は chordmap.json から読み込まれた辞書 (またはNone)
        
        # 1. emotion と intensity から基本パラメータを取得 (MAP参照)
        emotion_key = section_data.musical_intent.emotion
        intensity_key = section_data.musical_intent.intensity.lower()
        
        map_derived_params = NEW_EMOTION_TO_EXPRESSION_DETAILS_MAP.get(
            emotion_key, NEW_EMOTION_TO_EXPRESSION_DETAILS_MAP["default_expression_details"]
        ).copy()

        # 2. intensity に基づいて map_derived_params を調整
        if "high" in intensity_key:
            if map_derived_params.get("dynamics_profile") in ["mf", "f"]: map_derived_params["dynamics_profile"] = "ff"
            elif map_derived_params.get("dynamics_profile") == "mp": map_derived_params["dynamics_profile"] = "f"
            current_tempo_adjust = map_derived_params.get("tempo_feel_adjust_bpm", 0.0) or 0.0
            if current_tempo_adjust >= 0: map_derived_params["tempo_feel_adjust_bpm"] = current_tempo_adjust + 1.0
        elif "low" in intensity_key:
            if map_derived_params.get("dynamics_profile") in ["mf", "mp"]: map_derived_params["dynamics_profile"] = "p"
            elif map_derived_params.get("dynamics_profile") == "f": map_derived_params["dynamics_profile"] = "mp"
            current_tempo_adjust = map_derived_params.get("tempo_feel_adjust_bpm", 0.0) or 0.0
            if current_tempo_adjust <= 0: map_derived_params["tempo_feel_adjust_bpm"] = current_tempo_adjust - 1.0
        
        # 3. chordmap.json の expression_details があれば、それで map_derived_params を上書き
        #    (ただし、chordmap.json の expression_details には section_tonic/mode は含まれない想定)
        final_params_for_model = map_derived_params.copy() # MAP由来の値をベースにする
        if section_data.expression_details: # chordmap.json に expression_details があれば
            logger.debug(f"  Merging chordmap.json's expression_details: {section_data.expression_details}")
            for key, value in section_data.expression_details.items():
                if value is not None: # Noneでない値のみで上書き
                    final_params_for_model[key] = value
        
        # 4. ExpressionDetails Pydanticモデルを構築
        #    section_tonic と section_mode は Section モデルから持ってくる
        try:
            current_section_expression_details = ExpressionDetails(
                # section_tonic と section_mode は ExpressionDetails モデルから削除したので、ここでは不要
                recommended_tensions=final_params_for_model.get("recommended_tensions"),
                target_rhythm_category=final_params_for_model.get("target_rhythm_category"),
                approach_style=final_params_for_model.get("approach_style"),
                articulation_profile=final_params_for_model.get("articulation_profile"),
                humanize_profile=final_params_for_model.get("humanize_profile"),
                dynamics_profile=final_params_for_model.get("dynamics_profile"),
                tempo_feel_adjust_bpm=final_params_for_model.get("tempo_feel_adjust_bpm"),
                base_articulation=final_params_for_model.get("base_articulation"),
                voicing_style_piano_rh=final_params_for_model.get("voicing_style_piano_rh", section_data.part_settings.get("piano_rh_voicing_style")),
                voicing_style_piano_lh=final_params_for_model.get("voicing_style_piano_lh", section_data.part_settings.get("piano_lh_voicing_style")),
                voicing_style_guitar=final_params_for_model.get("voicing_style_guitar", section_data.part_settings.get("guitar_voicing_style")),
            )
        except ValidationError as e_expr_details:
            logger.error(f"  Error creating ExpressionDetails for section '{section_name}': {e_expr_details}")
            # フォールバックとしてデフォルトのExpressionDetailsを使うか、エラー処理
            current_section_expression_details = ExpressionDetails(
                 humanize_profile="default", dynamics_profile="mf", base_articulation="normal" # 最低限のフォールバック
            )
        # ▲▲▲ expression_details の生成ロジックここまで ▲▲▲

        logger.debug(f"  Final ExpressionDetails for section '{section_name}': {current_section_expression_details.model_dump(exclude_none=True)}")

        section_output = {
            "order": section_data.order,
            "length_in_measures": section_data.length_in_measures,
            "musical_intent": section_data.musical_intent.model_dump(),
            "expression_details": { # YAML出力用に section_tonic と section_mode を追加
                "section_tonic": section_data.tonic,
                "section_mode": section_data.mode,
                **current_section_expression_details.model_dump(exclude_none=True)
            },
            "part_settings": section_data.part_settings,
            "part_specific_hints": section_data.part_specific_hints,
            "processed_chord_events": []
        }

        if section_data.adjusted_start_beat is not None:
            current_absolute_offset_beats = section_data.adjusted_start_beat
            logger.info(f"  Adjusted start beat for section '{section_name}' to: {current_absolute_offset_beats}")

        section_relative_offset_beats = 0.0
        for chord_item in section_data.chord_progression:
            base_chord_label = chord_item.label
            base_duration = chord_item.duration_beats

            interpreted_details = get_interpreted_chord_details(
                base_chord_label,
                current_section_expression_details.recommended_tensions
            )
            final_chord_symbol_str = interpreted_details["interpreted_symbol"]
            specified_bass_str = interpreted_details["specified_bass"]

            if final_chord_symbol_str == "Rest":
                processed_event = {
                    "chord_symbol_for_voicing": "Rest", "specified_bass_for_voicing": None,
                    "original_duration_beats": base_duration, "original_offset_beats": round(section_relative_offset_beats, 4),
                    "humanized_duration_beats": base_duration, "humanized_offset_beats": round(section_relative_offset_beats, 4),
                }
            else:
                humanized_params = apply_emotional_expression_to_event(
                    base_duration_beats=base_duration,
                    base_offset_beats=section_relative_offset_beats,
                    humanize_profile_key=current_section_expression_details.humanize_profile or "default",
                    section_base_articulation=current_section_expression_details.base_articulation,
                    bpm=bpm + (current_section_expression_details.tempo_feel_adjust_bpm or 0.0),
                )
                processed_event = {
                    "chord_symbol_for_voicing": final_chord_symbol_str,
                    "specified_bass_for_voicing": specified_bass_str,
                    **humanized_params
                }
            
            processed_event["original_chord_label"] = base_chord_label
            processed_event["absolute_offset_beats"] = round(current_absolute_offset_beats + processed_event["humanized_offset_beats"], 4)
            section_output["processed_chord_events"].append(processed_event)
            section_relative_offset_beats += base_duration
        
        output_data["sections"][section_name] = section_output
        if section_data.adjusted_start_beat is None:
             current_absolute_offset_beats += section_relative_offset_beats

    try:
        with Path(output_yaml_path).open("w", encoding="utf-8") as f:
            yaml.safe_dump(output_data, f, allow_unicode=True, sort_keys=False, indent=2)
        logger.info(f"\nSuccessfully processed chordmap and wrote to: {output_yaml_path}")
    except Exception as e:
        print(f"Error writing output YAML to {output_yaml_path}: {e}")

# --- 6. CLI実行部分 --- (変更なし)
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process chordmap to apply emotional humanization.")
    parser.add_argument("input_json", type=str, help="Path to the input chordmap.json file.")
    parser.add_argument("output_yaml", type=str, help="Path for the output processed_chord_events.yaml file.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s: %(message)s')
    process_chordmap_for_emotion(args.input_json, args.output_yaml)
