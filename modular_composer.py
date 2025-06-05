import music21
from music21 import stream, tempo, meter, key, instrument as m21instrument, exceptions21
import copy
import math
import sys
import os
import json
import yaml
import argparse
import logging
import inspect
import random
from generator.base_part_generator import BasePartGenerator  # インポート
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, cast, Sequence

# --- ジェネレータのインポート ---
from generator.base_part_generator import BasePartGenerator
from generator import (  # 各ジェネレータを個別にインポート
    PianoGenerator,
    DrumGenerator,
    GuitarGenerator,
    ChordVoicer,
    MelodyGenerator,
    BassGenerator,
    VocalGenerator,
)

# --- ユーティリティとジェネレータのインポート ---
try:

    from utilities.rhythm_library_loader import (
        load_rhythm_library as load_rhythm_lib_main_func,
    )
    from utilities.override_loader import (
        load_overrides,
        get_part_override,
        Overrides,
        PartOverride,
    )
    from utilities.core_music_utils import get_time_signature_object
    from utilities.core_music_utils import (
        sanitize_chord_label,
    )  # ★ これを確実に使用する
    from utilities.scale_registry import ScaleRegistry, build_scale_object
    from utilities.humanizer import (
        apply_humanization_to_element,
        apply_humanization_to_part,
        HUMANIZATION_TEMPLATES,
        NUMPY_AVAILABLE,
    )
    from generator import (
        PianoGenerator,
        DrumGenerator,
        GuitarGenerator,
        ChordVoicer,
        MelodyGenerator,
        BassGenerator,
        VocalGenerator,
    )
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import modules: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
except Exception as e_imp:
    print(f"An unexpected error occurred during module import: {e_imp}")
    sys.exit(1)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - [%(levelname)s] - %(module)s.%(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("modular_composer")

logger.debug(f"Python sys.path: {sys.path}")
logger.debug(f"Current working directory: {os.getcwd()}")

DEFAULT_CONFIG = {
    "global_tempo": 120,
    "global_time_signature": "4/4",
    "global_key_tonic": "C",
    "global_key_mode": "major",
    "output_filename_template": "generated_song_{song_title}.mid",
    "parts_to_generate": {
        "piano": True,
        "drums": True,
        "bass": True,
        "guitar": True,
        "melody": False,
        "vocal": False,
        "chords": False,
    },
    "default_part_parameters": {
        "piano": {
            "instrument": "Piano",
            "default_humanize": False,
            "default_humanize_style_template": "piano_gentle_arpeggio",
            "default_rh_voicing_style": "spread_upper",
            "default_lh_voicing_style": "closed_low",
            "default_rh_target_octave": 4,
            "default_lh_target_octave": 2,
            "default_rh_num_voices": 4,
            "default_lh_num_voices": 2,
            "default_piano_rh_rhythm_key": "piano_fallback_block",
            "default_piano_lh_rhythm_key": "piano_lh_whole_notes",
            "default_velocity": 64,
        },
        "drums": {
            "instrument": "DrumSet",
            "default_humanize": False,
            "default_humanize_style_template": "drum_tight",
            "default_drum_style_key": "rock_beat_A_8th_hat",
            "default_drum_base_velocity": 80,
            "default_fill_interval_bars": 4,
            "default_fill_keys": ["rock_fill_1"],
        },
        "guitar": {
            "instrument": "AcousticGuitar",
            "default_humanize": False,
            "default_humanize_style_template": "guitar_strum_loose",
            "default_style": "strum_basic",
            "default_guitar_rhythm_key": "guitar_folk_strum_simple",
            "default_voicing_style": "standard_drop2",
            "default_num_strings": 6,
            "default_target_octave": 3,
            "default_velocity": 70,
        },
        "bass": {
            "instrument": "AcousticBass",
            "default_humanize": False,
            "default_humanize_style_template": "default_subtle",
            "default_rhythm_key": "bass_quarter_notes",
            "default_velocity": 80,
            "default_octave": 2,
            "default_weak_beat_style": "root",
            "default_options": {
                "approach_on_4th_beat": True,
                "approach_style_on_4th": "chromatic_or_diatonic",
            },
        },
        "melody": {
            "instrument": "Violin",
            "default_humanize": False,
            "default_rhythm_key": "melody_simple_quarters",
            "default_velocity": 70,
        },
        "vocal": {"instrument": "Voice", "default_humanize": False, "data_paths": {}},
        "chords": {
            "instrument": "ElectricPiano",
            "default_humanize": False,
            "default_velocity": 60,
        },
    },
    "rng_seed": None,
}


def load_json_file(file_path: Path, description: str) -> Optional[Dict | List]:
    if not file_path.exists():
        logger.error(f"{description} not found: {file_path}")
        sys.exit(1)  # Or raise an error
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Loaded {description} from: {file_path}")
        return data
    except json.JSONDecodeError as e_json:
        logger.error(
            f"Error decoding JSON from {description} at {file_path}: {e_json}",
            exc_info=True,
        )
        sys.exit(1)  # Or raise
    except Exception as e:
        logger.error(
            f"Error loading {description} from {file_path}: {e}", exc_info=True
        )
        sys.exit(1)  # Or raise


def load_yaml_file(file_path: Path, description: str) -> Optional[Dict]:
    if not file_path.exists():
        logger.error(f"{description} not found: {file_path}")
        sys.exit(1)  # Or raise an error
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        logger.info(f"Loaded {description} from: {file_path}")
        return data
    except yaml.YAMLError as e_yaml:
        logger.error(
            f"Error decoding YAML from {description} at {file_path}: {e_yaml}",
            exc_info=True,
        )
        sys.exit(1)  # Or raise
    except Exception as e:
        logger.error(
            f"Error loading {description} from {file_path}: {e}", exc_info=True
        )
        sys.exit(1)  # Or raise


def _get_humanize_params_for_final_touch(
    instrument_final_params: Dict[str, Any],
    default_cfg_instrument: Dict[str, Any],
) -> Dict[str, Any]:
    humanize_settings = {}
    final_touch_cfg = default_cfg_instrument.get("final_touch_humanize", {})
    humanize_settings["humanize_opt"] = final_touch_cfg.get("enable", False)
    if humanize_settings["humanize_opt"]:
        humanize_settings["template_name"] = final_touch_cfg.get(
            "template_name", "default_subtle"
        )
        humanize_settings["custom_params"] = final_touch_cfg.get("custom_params", {})
    return humanize_settings


def translate_and_merge_params_from_emotion_data(
    processed_event_data: Dict[
        str, Any
    ],  # これは emotion_humanizer からの event 単位のデータ
    section_part_settings: Dict[str, Any],  # chordmap のセクションごとの part_settings
    instrument_default_params_from_config: Dict[str, Any],  # DEFAULT_CONFIG の楽器設定
    instrument_name_key: str,
    rhythm_library_for_instrument: Dict[str, Any],  # あまり使われていない
    arrangement_override_for_part: Optional[PartOverride],  # アレンジオーバーライド
) -> Dict[str, Any]:
    final_params = instrument_default_params_from_config.copy()
    final_params.update(section_part_settings)  # セクション設定で上書き
    if arrangement_override_for_part:  # オーバーライドでさらに上書き
        override_dict = arrangement_override_for_part.model_dump(exclude_unset=True)
        if (
            "options" in override_dict
            and "options" in final_params
            and isinstance(final_params["options"], dict)
            and isinstance(override_dict["options"], dict)
        ):
            final_params["options"].update(override_dict.pop("options"))
        final_params.update(override_dict)

    # emotion_humanizer.py の出力 (processed_event_data) に含まれる emotion_params を適用
    emotion_params_in_event = processed_event_data.get(
        "emotion_params", {}
    )  # emotion_humanizer.py からの出力名
    # processed_event_data のトップレベルにも humanized_velocity などがあるのでそれも考慮
    # 優先順位: processed_event_data の humanized_velocity > emotion_params_in_event の velocity > final_params の velocity

    final_params["velocity"] = processed_event_data.get(
        "humanized_velocity",  # emotion_humanizer.py が直接設定した速度
        emotion_params_in_event.get("velocity", final_params.get("velocity")),
    )
    final_params["articulation_from_emotion"] = processed_event_data.get(
        "humanized_articulation",  # emotion_humanizer.py が直接設定したアーティキュレーション
        emotion_params_in_event.get("articulation"),
    )
    return final_params


def prepare_stream_for_generators(
    processed_chordmap_data: Dict,
    main_config: Dict,
    rhythm_lib_all: Dict,
    arrangement_overrides: Overrides,
) -> List[Dict]:
    logger.info("Preparing stream for generators from processed emotion chordmap...")
    stream_for_generators: List[Dict] = []
    global_settings = processed_chordmap_data.get("global_settings", {})
    sorted_sections = sorted(
        processed_chordmap_data.get("sections", {}).items(),
        key=lambda item: item[1].get("order", float("inf")),
    )

    for sec_name, sec_data in sorted_sections:
        sec_musical_intent = sec_data.get("musical_intent", {})
        sec_expression_details = sec_data.get("expression_details", {})
        sec_part_settings_overall = sec_data.get("part_settings", {})

        for event_idx, event_from_humanizer in enumerate(
            sec_data.get("processed_chord_events", [])
        ):
            original_label_from_yaml = event_from_humanizer.get("original_chord_label")
            # emotion_humanizer.py が出力した chord_symbol_for_voicing と specified_bass_for_voicing
            chord_for_voicing_from_yaml = event_from_humanizer.get(
                "chord_symbol_for_voicing"
            )
            bass_for_voicing_from_yaml = event_from_humanizer.get(
                "specified_bass_for_voicing"
            )

            # ▼▼▼ ここで sanitize_chord_label を適用 (最終防衛ライン) ▼▼▼
            final_sanitized_chord_symbol_for_generator: Optional[str] = (
                "Rest"  # デフォルトRest
            )
            final_sanitized_bass_for_generator: Optional[str] = None

            # 処理対象は emotion_humanizer.py が出力した `chord_for_voicing_from_yaml` と `bass_for_voicing_from_yaml`
            # `original_label_from_yaml` はあくまで元の記録用。

            temp_label_to_sanitize = chord_for_voicing_from_yaml

            if (
                temp_label_to_sanitize
                and str(temp_label_to_sanitize).strip().lower() != "rest"
            ):
                # ベース音が別指定されている場合、それを考慮してサニタイズ
                if bass_for_voicing_from_yaml and str(
                    bass_for_voicing_from_yaml
                ).strip().lower() not in ["", "rest", "r", "n.c.", "nc", "none"]:
                    # "C" と "Bb" -> "C/Bb" として sanitize_chord_label に渡す
                    combined_label = (
                        f"{temp_label_to_sanitize}/{bass_for_voicing_from_yaml}"
                    )
                    logger.debug(
                        f"ModComp (prepare_stream): Sanitizing combined '{combined_label}' (orig_yaml: '{chord_for_voicing_from_yaml}' / '{bass_for_voicing_from_yaml}')"
                    )
                    sanitized_result = sanitize_chord_label(combined_label)
                else:
                    # ベース音指定なし、または無効なベース音の場合はコード部分のみサニタイズ
                    logger.debug(
                        f"ModComp (prepare_stream): Sanitizing single '{temp_label_to_sanitize}' (orig_yaml: '{chord_for_voicing_from_yaml}', bass_yaml: '{bass_for_voicing_from_yaml}')"
                    )
                    sanitized_result = sanitize_chord_label(temp_label_to_sanitize)

                if sanitized_result and sanitized_result != "Rest":
                    # sanitize_chord_label が "C7/B-" のようにスラッシュ区切りで返すことを想定
                    if "/" in sanitized_result:
                        parts = sanitized_result.split("/", 1)
                        final_sanitized_chord_symbol_for_generator = parts[0]
                        final_sanitized_bass_for_generator = (
                            parts[1] if len(parts) > 1 else None
                        )
                    else:  # スラッシュなしで返ってきた場合 (例: "C7", "Am")
                        final_sanitized_chord_symbol_for_generator = sanitized_result
                        final_sanitized_bass_for_generator = (
                            None  # ベースは吸収されたか、元々ない
                        )
                else:  # サニタイズの結果 "Rest" または None になった
                    final_sanitized_chord_symbol_for_generator = "Rest"
                    final_sanitized_bass_for_generator = None
            else:  # 元からRest、または無効なコード指定
                final_sanitized_chord_symbol_for_generator = "Rest"
                final_sanitized_bass_for_generator = None

            logger.debug(
                f"  ModComp: Final for gen: Chord='{final_sanitized_chord_symbol_for_generator}', Bass='{final_sanitized_bass_for_generator}' (From YAML: Chord='{chord_for_voicing_from_yaml}', Bass='{bass_for_voicing_from_yaml}')"
            )
            # ▲▲▲ サニタイズ処理ここまで ▲▲▲

            blk_data = {
                "absolute_offset": event_from_humanizer.get("absolute_offset_beats"),
                "q_length": event_from_humanizer.get("humanized_duration_beats"),
                "original_chord_label": original_label_from_yaml,  # 元のラベルは保持
                "chord_symbol_for_voicing": final_sanitized_chord_symbol_for_generator,  # ★最終サニタイズ済み
                "specified_bass_for_voicing": final_sanitized_bass_for_generator,  # ★最終サニタイズ済み
                "section_name": sec_name,
                "tonic_of_section": sec_expression_details.get("section_tonic"),
                "mode": sec_expression_details.get("section_mode"),
                "is_first_in_section": (event_idx == 0),
                "is_last_in_section": (
                    event_idx == len(sec_data.get("processed_chord_events", [])) - 1
                ),
                "musical_intent": sec_musical_intent,
                "expression_details": sec_expression_details,
                "emotion_params": event_from_humanizer.get(
                    "emotion_profile_applied", {}
                ),  # emotion_humanizer.pyからの出力
                "part_params": {},
            }

            # 各パートのパラメータ設定
            for part_name, generate_flag in main_config.get(
                "parts_to_generate", {}
            ).items():
                if generate_flag:
                    instrument_default_cfg = main_config["default_part_parameters"].get(
                        part_name, {}
                    )
                    rhythm_category_key = None
                    if part_name == "drums":
                        rhythm_category_key = "drum_patterns"
                    elif part_name == "bass":
                        rhythm_category_key = "bass_patterns"
                    elif part_name == "piano":
                        rhythm_category_key = "piano_patterns"
                    elif part_name == "guitar":
                        rhythm_category_key = "guitar_patterns"
                    elif part_name == "melody":
                        rhythm_category_key = "melody_rhythms"

                    rhythm_lib_for_instrument = (
                        rhythm_lib_all.get(rhythm_category_key, {})
                        if rhythm_category_key
                        else {}
                    )
                    part_override_model = get_part_override(
                        arrangement_overrides, sec_name, part_name
                    )

                    # event_from_humanizer には emotion_profile_applied が含まれるので、それを渡す
                    # translate_and_merge_params_from_emotion_data は humanized_velocity と humanized_articulation も見る
                    final_instrument_params = translate_and_merge_params_from_emotion_data(
                        processed_event_data=event_from_humanizer,  # emotion_humanizer からのイベントデータ全体
                        section_part_settings=sec_part_settings_overall.get(
                            part_name, {}
                        ),
                        instrument_default_params_from_config=instrument_default_cfg,
                        instrument_name_key=part_name,
                        rhythm_library_for_instrument=rhythm_lib_for_instrument,
                        arrangement_override_for_part=part_override_model,
                    )
                    # blk_data["emotion_params"] には emotion_profile_applied が入っているので、
                    # final_instrument_params["velocity"] などはそれを元に設定される。
                    # humanized_velocity も translate_and_merge_params_from_emotion_data 内で考慮される。
                    blk_data["part_params"][part_name] = final_instrument_params
            stream_for_generators.append(blk_data)
    logger.info(f"Prepared {len(stream_for_generators)} blocks for generators.")
    return stream_for_generators


# ... (DEFAULT_CONFIG, load_json_file, load_yaml_file, _get_humanize_params_for_final_touch,
#      translate_and_merge_params_from_emotion_data, prepare_stream_for_generators は変更なし) ...


def run_composition(
    cli_args: argparse.Namespace,
    main_cfg: Dict,
    processed_chordmap_data: Dict,
    rhythm_lib_data: Dict,
    arrangement_overrides: Optional["Overrides"] = None,
):
    g_settings_proc = processed_chordmap_data.get("global_settings", {})
    global_tempo_val = g_settings_proc.get("tempo", main_cfg["global_tempo"])
    global_ts_str = g_settings_proc.get(
        "time_signature", main_cfg["global_time_signature"]
    )
    global_key_tonic_val = g_settings_proc.get(
        "key_tonic", main_cfg["global_key_tonic"]
    )
    global_key_mode_val = g_settings_proc.get("key_mode", main_cfg["global_key_mode"])
    final_score = stream.Score()
    final_score.insert(0, tempo.MetronomeMark(number=global_tempo_val))
    ts_obj_score = get_time_signature_object(global_ts_str)
    final_score.insert(0, ts_obj_score if ts_obj_score else meter.TimeSignature("4/4"))
    final_score.insert(
        0, key.Key(global_key_tonic_val, str(global_key_mode_val).lower())
    )

    proc_blocks = prepare_stream_for_generators(
        processed_chordmap_data, main_cfg, rhythm_lib_data, arrangement_overrides
    )
    if not proc_blocks:
        logger.error("No blocks to process. Aborting.")
        return

    # ▼▼▼ gens 変数の初期化とジェネレータインスタンスの格納 ▼▼▼
    gens: Dict[str, BasePartGenerator] = {}  # 型ヒントを BasePartGenerator に
    cv_inst = ChordVoicer(
        global_tempo=global_tempo_val, global_time_signature=global_ts_str
    )  # ChordVoicerは別途

    for part_name, generate_flag in main_cfg.get("parts_to_generate", {}).items():
        if not generate_flag:
            continue

        part_default_cfg = main_cfg["default_part_parameters"].get(part_name, {})
        instrument_str = part_default_cfg.get("instrument", "Piano")
        try:
            instrument_obj = m21instrument.fromString(instrument_str)
        except Exception:
            instrument_obj = m21instrument.Piano()

        rhythm_category_key: Optional[str] = None
        if part_name == "drums":
            rhythm_category_key = "drum_patterns"
        elif part_name == "bass":
            rhythm_category_key = "bass_patterns"
        elif part_name == "piano":
            rhythm_category_key = "piano_patterns"
        elif part_name == "guitar":
            rhythm_category_key = "guitar_patterns"
        elif part_name == "melody":
            rhythm_category_key = "melody_rhythms"

        rhythm_lib_for_instrument: Dict[str, Any] = (
            rhythm_lib_data.get(rhythm_category_key, {}) if rhythm_category_key else {}
        )

        # BasePartGenerator を継承するジェネレータのインスタンス化
        if part_name == "bass":
            gens[part_name] = BassGenerator(
                part_name=part_name,
                part_parameters=rhythm_lib_for_instrument,  # ← 引数名をBasePartGeneratorに合わせる
                main_cfg=main_cfg,
                groove_profile=None,  # 必要に応じて
                global_tempo=global_tempo_val,
                global_time_signature_obj=ts_obj_score,
            )
        # --- 他のジェネレータも同様に BasePartGenerator を継承し、ここでインスタンス化 ---
        elif part_name == "piano":
            logger.warning(
                f"PianoGenerator is not yet fully adapted to BasePartGenerator. Skipping {part_name}."
            )
            pass
        elif part_name == "drums":
            logger.warning(
                f"DrumGenerator is not yet fully adapted to BasePartGenerator. Skipping {part_name}."
            )
            pass
        elif part_name == "guitar":
            logger.warning(
                f"GuitarGenerator is not yet fully adapted to BasePartGenerator. Skipping {part_name}."
            )
            pass
        # ... (MelodyGenerator, VocalGenerator も同様) ...
        elif part_name == "chords":  # ChordVoicer は BasePartGenerator を継承しない
            if instrument_obj:
                cv_inst.default_instrument = instrument_obj
            # ChordVoicer は gens には含めず、ループの外で別途処理する
    # ▲▲▲ gens 変数の初期化とジェネレータインスタンスの格納ここまで ▲▲▲

    # パート生成ループ
    for i, current_block_data in enumerate(proc_blocks):
        next_block_data = proc_blocks[i + 1] if i + 1 < len(proc_blocks) else None

        # "chords" パートの処理 (ChordVoicer を直接使用)
        if main_cfg["parts_to_generate"].get("chords"):
            logger.info(f"Generating chords part using ChordVoicer for block {i+1}...")
            try:
                chord_part_obj = cv_inst.compose([current_block_data])
                if (
                    isinstance(chord_part_obj, stream.Part)
                    and chord_part_obj.flatten().notesAndRests
                ):
                    final_score.insert(
                        current_block_data["absolute_offset"], chord_part_obj
                    )
            except Exception as e_cv_compose:
                logger.error(
                    f"Error in ChordVoicer compose for block {i+1}: {e_cv_compose}",
                    exc_info=True,
                )

        for part_name, generator_instance in gens.items():
            if generator_instance and main_cfg["parts_to_generate"].get(part_name):
                logger.info(
                    f"Generating {part_name} part for block {i+1} via BasePartGenerator..."
                )
                try:
                    part_humanize_cfg = main_cfg["default_part_parameters"].get(
                        part_name, {}
                    )
                    humanize_params_for_part = {
                        "humanize_opt": current_block_data["part_params"][
                            part_name
                        ].get(
                            "humanize_opt",
                            part_humanize_cfg.get("default_humanize", False),
                        ),
                        "template_name": current_block_data["part_params"][
                            part_name
                        ].get(
                            "template_name",
                            part_humanize_cfg.get("default_humanize_style_template"),
                        ),
                        "custom_params": current_block_data["part_params"][
                            part_name
                        ].get("custom_params", {}),
                    }

                    part_obj = generator_instance.compose(
                        section_data=current_block_data,
                        overrides_root=arrangement_overrides,
                        groove_profile_path=(
                            cli_args.groove_profile
                            if hasattr(cli_args, "groove_profile")
                            and cli_args.groove_profile
                            else None
                        ),
                        next_section_data=next_block_data,
                        part_specific_humanize_params=humanize_params_for_part,
                    )

                    if (
                        isinstance(part_obj, stream.Part)
                        and part_obj.flatten().notesAndRests
                    ):
                        final_score.insert(
                            current_block_data["absolute_offset"], part_obj
                        )
                    elif isinstance(part_obj, stream.Score) and part_obj.parts:
                        for sub_part in part_obj.parts:
                            if sub_part.flatten().notesAndRests:
                                final_score.insert(
                                    current_block_data["absolute_offset"], sub_part
                                )
                except Exception as e_gen_part:
                    logger.error(
                        f"Error in {part_name} generation for block {i+1}: {e_gen_part}",
                        exc_info=True,
                    )

    # ここでMIDIファイル出力
    title = (
        processed_chordmap_data.get("project_title", "untitled")
        .replace(" ", "_")
        .lower()
    )
    out_fname_template = main_cfg.get(
        "output_filename_template", "output_{song_title}.mid"
    )
    actual_out_fname = (
        cli_args.output_filename
        if cli_args.output_filename
        else out_fname_template.format(song_title=title)
    )
    out_fpath = cli_args.output_dir / actual_out_fname
    out_fpath.parent.mkdir(parents=True, exist_ok=True)
    try:
        if final_score.flatten().notesAndRests:
            final_score.write("midi", fp=str(out_fpath))
            logger.info(f"🎉 MIDI exported to {out_fpath}")
        else:
            logger.warning(f"Score is empty. No MIDI file generated at {out_fpath}.")
    except music21.exceptions21.Music21Exception as e_m21w:
        logger.error(
            f"Music21 MIDI write error to {out_fpath}: {e_m21w}", exc_info=True
        )
    except Exception as e_w:
        logger.error(f"General MIDI write error to {out_fpath}: {e_w}", exc_info=True)


def main_cli():
    # (ArgumentParser の設定は変更なし)
    parser = argparse.ArgumentParser(
        description="Modular Music Composer with Emotion Humanizer"
    )
    parser.add_argument(
        "processed_chordmap_file",
        type=Path,
        help="Path to the processed_chordmap_with_emotion.yaml file.",
    )
    parser.add_argument(
        "rhythm_library_file",
        type=Path,
        help="Path to the rhythm library (JSON/YAML/TOML) file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("midi_output"),
        help="Directory to save the output MIDI file.",
    )
    parser.add_argument(
        "--output-filename", type=str, help="Custom filename for the output MIDI file."
    )
    parser.add_argument(
        "--settings-file",
        type=Path,
        help="Path to a custom settings JSON file to override defaults.",
    )
    parser.add_argument(
        "--tempo",
        type=int,
        help="Override global tempo defined in processed chordmap or DEFAULT_CONFIG.",
    )
    parser.add_argument(
        "--vocal-mididata-path",
        type=str,
        help="Path to vocal MIDI data JSON (overrides config).",
    )
    parser.add_argument(
        "--rng-seed",
        type=int,
        help="Seed for random number generator for reproducibility.",
    )
    parser.add_argument(
        "--overrides-file",
        type=Path,
        help="Path to the arrangement overrides JSON file.",
    )
    parser.add_argument(
        "--guitar-style",
        type=str,
        help="Override guitar style/rhythm key for the entire song.",
    )
    parser.add_argument(
        "--groove-profile", type=str, help="Path to groove profile JSON file."
    )  # groove_profile引数を追加

    default_parts_cfg = DEFAULT_CONFIG.get("parts_to_generate", {})
    for part_key, default_enabled_status in default_parts_cfg.items():
        arg_name_for_part = f"generate_{part_key}"
        if default_enabled_status:
            parser.add_argument(
                f"--no-{part_key}",
                action="store_false",
                dest=arg_name_for_part,
                help=f"Disable {part_key} generation.",
            )
        else:
            parser.add_argument(
                f"--include-{part_key}",
                action="store_true",
                dest=arg_name_for_part,
                help=f"Enable {part_key} generation.",
            )
    arg_defaults = {f"generate_{k}": v for k, v in default_parts_cfg.items()}
    parser.set_defaults(**arg_defaults)
    args = parser.parse_args()

    # (effective_cfg の設定、ファイルロードなどは変更なし)
    effective_cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    if args.settings_file and args.settings_file.exists():
        custom_settings_data = load_json_file(args.settings_file, "Custom settings")
        if custom_settings_data and isinstance(custom_settings_data, dict):

            def _deep_update(target_dict, source_dict):
                for key_item, value_item in source_dict.items():
                    if (
                        isinstance(value_item, dict)
                        and key_item in target_dict
                        and isinstance(target_dict[key_item], dict)
                    ):
                        _deep_update(target_dict[key_item], value_item)
                    else:
                        target_dict[key_item] = value_item

            _deep_update(effective_cfg, custom_settings_data)
            logger.info(f"Loaded and merged custom settings from {args.settings_file}")
    for pk_name in default_parts_cfg.keys():
        arg_name_cli = f"generate_{pk_name}"
        if hasattr(args, arg_name_cli) and getattr(args, arg_name_cli) is not None:
            effective_cfg["parts_to_generate"][pk_name] = getattr(args, arg_name_cli)
    if args.vocal_mididata_path:
        if "vocal" not in effective_cfg["default_part_parameters"]:
            effective_cfg["default_part_parameters"]["vocal"] = {}
        if "data_paths" not in effective_cfg["default_part_parameters"]["vocal"]:
            effective_cfg["default_part_parameters"]["vocal"]["data_paths"] = {}
        effective_cfg["default_part_parameters"]["vocal"]["data_paths"][
            "midivocal_data_path"
        ] = str(args.vocal_mididata_path)
    if args.rng_seed is not None:
        effective_cfg["rng_seed"] = args.rng_seed
        random.seed(args.rng_seed)
        logger.info(f"RNG Seed set to: {args.rng_seed}")

    processed_chordmap_data_loaded = load_yaml_file(
        args.processed_chordmap_file, "Processed Chordmap with Emotion"
    )
    if not processed_chordmap_data_loaded:
        sys.exit(1)
    logger.info(f"Loading rhythm library from: {args.rhythm_library_file}")
    try:
        rhythm_library_model = load_rhythm_lib_main_func(args.rhythm_library_file)
        if not rhythm_library_model:
            logger.critical(
                f"Failed to load rhythm library from {args.rhythm_library_file} (loader returned None). Exit."
            )
            sys.exit(1)
        rhythm_library_data_loaded = rhythm_library_model.model_dump()
        logger.info(
            f"Rhythm library loaded and parsed from {args.rhythm_library_file}."
        )
    except FileNotFoundError:
        logger.critical(
            f"Rhythm library file not found: {args.rhythm_library_file}. Exit."
        )
        sys.exit(1)
    except ValueError as e_val:
        logger.critical(
            f"Error processing rhythm library {args.rhythm_library_file}: {e_val}. Exit.",
            exc_info=False,
        )
        sys.exit(1)
    except Exception as e_rhythm_load:
        logger.critical(
            f"Unexpected error loading rhythm library {args.rhythm_library_file}: {e_rhythm_load}. Exit.",
            exc_info=True,
        )
        sys.exit(1)
    if not rhythm_library_data_loaded or not isinstance(
        rhythm_library_data_loaded, dict
    ):
        logger.critical("Rhythm library data invalid after loading. Exit.")
        sys.exit(1)

    cm_globals_loaded = processed_chordmap_data_loaded.get("global_settings", {})
    effective_cfg["global_tempo"] = cm_globals_loaded.get(
        "tempo", effective_cfg["global_tempo"]
    )
    effective_cfg["global_time_signature"] = cm_globals_loaded.get(
        "time_signature", effective_cfg["global_time_signature"]
    )
    effective_cfg["global_key_tonic"] = cm_globals_loaded.get(
        "key_tonic", effective_cfg["global_key_tonic"]
    )
    effective_cfg["global_key_mode"] = cm_globals_loaded.get(
        "key_mode", effective_cfg["global_key_mode"]
    )
    if args.tempo is not None:
        effective_cfg["global_tempo"] = args.tempo
        logger.info(f"Global tempo overridden by CLI to: {args.tempo}")

    # ▼▼▼ arrangement_overrides のロード処理をここに追加 ▼▼▼
    arrangement_overrides: Overrides = Overrides(root={})  # デフォルト初期化
    override_file_path_to_load: Optional[Path] = None

    if args.overrides_file:
        override_file_path_to_load = args.overrides_file
    elif Path("data/arrangement_overrides.json").exists():
        override_file_path_to_load = Path("data/arrangement_overrides.json")
    elif Path("data/arrangement_overrides.yaml").exists():
        override_file_path_to_load = Path("data/arrangement_overrides.yaml")
    elif Path("data/arrangement_overrides.yml").exists():
        override_file_path_to_load = Path("data/arrangement_overrides.yml")

    if override_file_path_to_load:
        logger.info(f"Attempting to load overrides from: {override_file_path_to_load}")
        try:
            arrangement_overrides = load_overrides(str(override_file_path_to_load))
            logger.info(
                f"Successfully loaded arrangement overrides from: {override_file_path_to_load}"
            )
        except Exception as e_load_ov:
            logger.error(
                f"Error loading overrides file {override_file_path_to_load}: {e_load_ov}. Proceeding without overrides."
            )
            arrangement_overrides = Overrides(root={})
    else:
        logger.info(
            "No overrides file specified or found at default locations. Proceeding without overrides."
        )
    # ▲▲▲ arrangement_overrides のロード処理ここまで ▲▲▲

    logger.info(
        f"Final Effective Config (using processed chordmap): {json.dumps(effective_cfg, indent=2, ensure_ascii=False)}"
    )

    try:
        run_composition(
            args,
            effective_cfg,
            processed_chordmap_data_loaded,
            rhythm_library_data_loaded,
            arrangement_overrides,  # ★ main_cli スコープで定義されたものを渡す
        )
    except SystemExit:
        raise
    except Exception as e_main_run:
        logger.critical(f"Critical error in main run: {e_main_run}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main_cli()
# --- END OF FILE modular_composer.py ---
