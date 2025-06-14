# utilities/rhythm_library_loader.py (Pydanticモデル修正版)
from __future__ import annotations

import json
import logging
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Final, List, Literal, Optional, Union

import yaml  # type: ignore
import tomli  # type: ignore
from pydantic import BaseModel, Field, ValidationError, field_validator

LOGGER = logging.getLogger(__name__)


class PatternEvent(BaseModel):
    offset: float = Field(
        ...,
        alias="beat",  # YAMLの "beat" キーを "offset" フィールドにマッピング
        description="Offset from the start of the pattern unit in quarter lengths.",
        ge=0,
    )
    duration: float = Field(
        1.0,
        description="Duration of the event in quarter lengths.",
        ge=0.0,
    )
    velocity: Optional[int] = Field(
        None,
        description="Absolute velocity (1-127). Overrides velocity_factor if present.",
        ge=1,
        le=127,
    )
    velocity_factor: Optional[float] = Field(
        None,
        description="Velocity multiplier (0.0 to 1.0+). Used if velocity is not set.",
        ge=0,
    )
    instrument: Optional[str] = Field(
        None,
        description="Specific instrument for multi-instrument parts (e.g., drums).",
    )
    type: Optional[str] = Field(
        None,
        description="Type of note (e.g., 'root', 'fifth', 'ghost') for algorithmic hints.",
    )
    strum_direction: Optional[str] = Field(
        None, description="For guitar: 'up' or 'down'."
    )
    scale_degree: Optional[Union[int, str]] = Field(
        None, description="For basslines: scale degree to target."
    )
    octave: Optional[int] = Field(
        None, description="For basslines: specific octave for the note."
    )
    glide_to_next: Optional[bool] = Field(
        None, description="For basslines: if the note should glide to the next."
    )
    accent: Optional[bool] = Field(None, description="If the event should be accented.")
    probability: Optional[float] = Field(
        1.0,
        description="Probability of this event occurring (0.0 to 1.0).",
        ge=0.0,
        le=1.0,
    )
    model_config = {
        "extra": "allow",
        "populate_by_name": True,
    }  # populate_by_name を True に


class BasePatternDef(BaseModel):
    description: Optional[str] = None
    tags: Optional[List[str]] = Field(default_factory=list)
    time_signature: Optional[str] = "4/4"
    length_beats: Optional[float] = (
        Field(  # リズムライブラリYAMLの length_beats と合わせる
            4.0, description="Reference length of the pattern in beats."
        )
    )
    reference_duration_ql: Optional[float] = (
        Field(  # ジェネレータが参照する可能性のあるキー
            None,
            description="Reference duration of the pattern in quarter lengths. If None, often calculated from length_beats and time_signature.",
        )
    )
    pattern_type: Optional[str] = "fixed_pattern"
    velocity_base: Optional[int] = Field(None, ge=1, le=127)
    options: Optional[Dict[str, Any]] = Field(default_factory=dict)
    pattern: Optional[List[PatternEvent]] = None
    model_config = {"extra": "allow"}

    @field_validator("reference_duration_ql", mode="before")
    @classmethod
    def calculate_reference_duration_ql(cls, v, values):
        if v is None:
            data = values.data  # Pydantic v2
            length_beats = data.get("length_beats", 4.0)
            time_signature_str = data.get("time_signature", "4/4")
            try:
                # music21を使って拍子記号から1拍の長さを取得し、全体の長さを計算
                # このバリデータ内でmusic21をインポートするのは重いので、単純計算に留めるか、
                # または、このフィールドはオプショナルとし、利用側で計算する。
                # ここでは単純に length_beats を quarterLength とみなす（4/4拍子なら等価）
                # より正確には time_signature をパースする必要がある。
                # 例: 4/4なら1拍=1.0ql, 3/4なら1拍=1.0ql, 6/8なら1拍=0.5ql (付点4分音符が1拍の場合)
                # 簡単のため、length_beats をそのまま ql とする。
                # 必要なら、get_time_signature_object を使って計算する。
                if time_signature_str == "4/4":  # 簡単な例
                    return length_beats
                # 他の拍子の場合は、より正確な計算が必要
                ts_obj = meter.TimeSignature(
                    time_signature_str
                )  # music21のmeterをインポート必要
                return length_beats * ts_obj.beatDuration.quarterLength

            except Exception:
                return length_beats  # フォールバック
        return v


class PianoPattern(BasePatternDef):
    arpeggio_type: Optional[str] = None
    note_duration_ql: Optional[float] = None


class DrumPattern(BasePatternDef):
    swing: Optional[Union[float, Dict[str, Any]]] = None
    fill_ins: Optional[Dict[str, List[PatternEvent]]] = Field(default_factory=dict)
    inherit: Optional[str] = None


class BassPattern(BasePatternDef):
    target_octave: Optional[int] = None
    note_duration_ql: Optional[float] = None


class GuitarPattern(BasePatternDef):
    pattern: Optional[Union[List[PatternEvent], List[float], List[int]]] = (
        None  # PatternEvent以外は非推奨
    )
    execution_style: Optional[str] = None  # ★★★ 追加 ★★★
    arpeggio_indices: Optional[List[int]] = None
    strum_width_sec: Optional[float] = None
    note_duration_ql: Optional[float] = None
    step_duration_ql: Optional[float] = None
    note_articulation_factor: Optional[float] = None
    strum_direction_cycle: Optional[List[str]] = None
    tremolo_rate_hz: Optional[int] = None
    crescendo_curve: Optional[str] = None
    duration_beats: Optional[float] = None  # length_beats と重複する可能性あるので注意
    velocity_start: Optional[int] = Field(None, ge=1, le=127)
    velocity_end: Optional[int] = Field(None, ge=1, le=127)
    palm_mute_level_recommended: Optional[float] = Field(None, ge=0.0, le=1.0)


class RhythmLibrary(BaseModel):
    piano_patterns: Optional[Dict[str, PianoPattern]] = Field(default_factory=dict)
    drum_patterns: Optional[Dict[str, DrumPattern]] = Field(default_factory=dict)
    bass_patterns: Optional[Dict[str, BassPattern]] = Field(default_factory=dict)
    guitar: Optional[Dict[str, GuitarPattern]] = (
        Field(  # ★★★ guitar_patterns から guitar に変更 ★★★
            default_factory=dict,
            alias="guitar_patterns",  # YAMLでは guitar_patterns を期待
        )
    )
    extra: Dict[str, Any] = Field(
        default_factory=dict
    )  # その他のトップレベルキーを許容
    model_config = {
        "extra": "allow",
        "str_max_length": 2048,
        "populate_by_name": True,
    }  # populate_by_name を True に


LIB_PATH_ENV: Final[str] = "RHYTHM_LIBRARY_PATH"
EXTRA_DIR_ENV: Final[str] = "RHYTHM_EXTRA_DIR"


@lru_cache(maxsize=None)
def load_rhythm_library(
    path: str | os.PathLike[str] | None = None,
    *,
    extra_dir: str | os.PathLike[str] | None = None,
    force_reload: bool = False,
) -> RhythmLibrary:
    if force_reload:
        load_rhythm_library.cache_clear()

    src_path = _resolve_main_path(path)
    raw = _parse_file(src_path)  # ここではまだPythonのdict

    extra_dir_resolved = _resolve_extra_dir(extra_dir)
    if extra_dir_resolved:
        raw = _merge_extra_patterns(raw, extra_dir_resolved)

    for name, pat in raw.get("drum_patterns", {}).items():
        for ev in pat.get("pattern", []):
            if "duration" not in ev:
                ev["duration"] = 1.0

    try:
        # Pydanticモデルによるバリデーションと型変換
        # ここで alias="guitar_patterns" が機能し、raw["guitar_patterns"] が lib.guitar にマッピングされる
        lib = RhythmLibrary.model_validate(raw)
    except ValidationError as exc:
        error_details = _format_pydantic_errors(exc)
        LOGGER.error(
            f"Rhythm library validation failed for {src_path}:\n{error_details}"
        )
        raise ValueError(
            f"Rhythm library validation failed for {src_path}:\n{error_details}"
        )

    LOGGER.info(
        "Rhythm library loaded: %s (+%s extras)",
        src_path.name,
        len(list(extra_dir_resolved.glob("*/*"))) if extra_dir_resolved else 0,
    )
    return lib


def _resolve_main_path(path: str | os.PathLike[str] | None) -> Path:
    if path is None:
        path_str = os.getenv(LIB_PATH_ENV)
        if not path_str:
            path_str = "data/rhythm_library.yml"  # デフォルトパス
            LOGGER.info(f"Rhythm library path not specified, using default: {path_str}")
        path = path_str
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Rhythm library not found: {p}")
    return p


def _resolve_extra_dir(extra_dir: str | os.PathLike[str] | None) -> Path | None:
    if extra_dir is None:
        extra_dir_str = os.getenv(EXTRA_DIR_ENV)
        if not extra_dir_str:
            extra_dir_str = "extra_patterns"  # デフォルトのextraディレクトリ
            # LOGGER.info(f"Rhythm extra directory not specified, using default: {extra_dir_str}")
        extra_dir = extra_dir_str
    p = Path(extra_dir).expanduser().resolve()
    if p.exists() and p.is_dir():
        return p
    else:
        # LOGGER.info(f"Rhythm extra directory not found or not a directory: {p}")
        return None


def _parse_file(path: Path) -> Dict[str, Any]:
    LOGGER.debug(f"Parsing rhythm library file: {path}")
    try:
        with path.open("r", encoding="utf-8") as f:
            content = f.read()
            if not content.strip():  # 空ファイルの場合
                LOGGER.warning(f"Rhythm library file is empty: {path}")
                return {}
            if path.suffix.lower() == ".json":
                return json.loads(content)
            elif path.suffix.lower() in {".yaml", ".yml"}:
                return yaml.safe_load(content)
            elif path.suffix.lower() == ".toml":
                return tomli.loads(content)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")
    except Exception as e:
        LOGGER.error(f"Error parsing file {path}: {e}", exc_info=True)
        raise


def _merge_extra_patterns(base: Dict[str, Any], extra_dir: Path) -> Dict[str, Any]:
    merged = base.copy()
    for file_path in extra_dir.rglob("*.*"):  # サブディレクトリも再帰的に検索
        if file_path.suffix.lower() not in {".json", ".yaml", ".yml", ".toml"}:
            continue  # サポートされていないファイル形式はスキップ
        if file_path.is_file():
            try:
                LOGGER.debug(f"Merging extra patterns from: {file_path}")
                extra_data = _parse_file(file_path)
                for top_key, category_data in extra_data.items():
                    if (
                        top_key in merged
                        and isinstance(merged[top_key], dict)
                        and isinstance(category_data, dict)
                    ):
                        # カテゴリレベルでディープマージ（パターンキーが重複したら上書き）
                        for pattern_key, pattern_value in category_data.items():
                            merged[top_key][pattern_key] = pattern_value
                    elif isinstance(
                        category_data, dict
                    ):  # ベースにないカテゴリならそのまま追加
                        merged[top_key] = category_data
                    else:
                        LOGGER.warning(
                            f"Cannot merge data for key '{top_key}' from {file_path}, as it's not a dictionary."
                        )
            except Exception as exc:
                LOGGER.warning(
                    f"Skipping extra pattern file {file_path.name} due to error: {exc}"
                )
                continue
    return merged


def _format_pydantic_errors(exc: ValidationError) -> str:
    lines = []
    for error in exc.errors():
        loc_str = " -> ".join(str(loc_item) for loc_item in error["loc"])
        lines.append(f"  - Location: {loc_str}")
        lines.append(f"    Message: {error['msg']}")
        lines.append(f"    Type: {error['type']}")
        if "input" in error:
            input_val = error["input"]
            # 入力値が長すぎる場合に省略する
            input_str = str(input_val)
            if len(input_str) > 200:  # 例えば200文字以上なら省略
                input_str = input_str[:200] + "..."
            lines.append(f"    Input: {input_str}")
        if "ctx" in error and error["ctx"]:
            lines.append(f"    Context: {error['ctx']}")
        lines.append("")  # 各エラーの間に空行
    return "\n".join(lines)


if __name__ == "__main__":
    # (CLI部分は変更なし)
    # music21.meterをインポート (BasePatternDefのバリデータで使用するため)
    from music21 import meter as m21_meter  # BasePatternDefのバリデータで使用

    import argparse
    import pprint

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Validate and inspect rhythm_library file"
    )
    parser.add_argument(
        "library_file",
        nargs="?",
        help="Path to rhythm_library file (JSON, YAML, or TOML). Uses RHYTHM_LIBRARY_PATH or default if not given.",
    )
    parser.add_argument(
        "--extra-dir",
        help="Directory containing extra pattern files. Uses RHYTHM_EXTRA_DIR or default if not given.",
    )
    parser.add_argument(
        "--list",
        choices=["piano", "drums", "bass", "guitar"],
        help="List pattern keys of a specific category.",
    )
    parser.add_argument(
        "--show",
        metavar="PATH.TO.KEY",
        help="Show pattern detail (dot‑separated path, e.g., guitar.guitar_ballad_arpeggio).",  # guitar_patternsからguitarに変更
    )
    parser.add_argument(
        "--force-reload",
        action="store_true",
        help="Force reload the library, bypassing cache.",
    )
    args = parser.parse_args()
    try:
        lib_model = load_rhythm_library(
            args.library_file, extra_dir=args.extra_dir, force_reload=args.force_reload
        )
        print("Rhythm library loaded and validated successfully! 🎉")
        if args.list:
            # RhythmLibraryモデルのフィールド名に合わせてアクセス
            category_name_in_model = args.list
            if args.list != "guitar":  # guitar以外は末尾に_patternsがつく
                category_name_in_model = f"{args.list}_patterns"

            # lib_model.guitar のようにアクセスするため、args.list を直接使う
            category_data = getattr(
                lib_model, args.list, None
            )  # piano, drums, bass, guitar

            if category_data:
                print(f"\nPatterns in '{args.list}':")  # 表示は元のカテゴリ名で
                if isinstance(category_data, dict):
                    pprint.pprint(list(category_data.keys()))
                else:
                    print(f"Category '{args.list}' is not a dictionary in the model.")
            else:
                print(f"Category '{args.list}' not found in the library model.")
        elif args.show:
            path_parts = args.show.split(".")
            current_node: Any = lib_model
            valid_path = True
            for part_idx, part_name in enumerate(path_parts):
                if isinstance(current_node, BaseModel):
                    # Pydanticモデルのフィールドとしてアクセス試行
                    if hasattr(current_node, part_name):
                        current_node = getattr(current_node, part_name)
                    else:  # フィールドにない場合、model_dumpしてから辞書としてアクセス試行
                        dumped_node = current_node.model_dump()
                        if isinstance(dumped_node, dict) and part_name in dumped_node:
                            current_node = dumped_node[part_name]
                        else:
                            valid_path = False
                            break
                elif isinstance(current_node, dict):
                    if part_name in current_node:
                        current_node = current_node[part_name]
                    else:
                        valid_path = False
                        break
                else:  # それ以外（リストやプリミティブ型など）でさらにパスが続く場合は無効
                    valid_path = False
                    break

            if valid_path:
                print(f"\nDetails for '{args.show}':")
                if isinstance(current_node, BaseModel):  # 表示はmodel_dumpで
                    pprint.pprint(current_node.model_dump(exclude_unset=True))
                else:
                    pprint.pprint(current_node)
            else:
                print(f"Path '{args.show}' not found in the library model.")

    except FileNotFoundError as e:
        LOGGER.error(f"File not found: {e}")
        sys.exit(1)
    except ValueError as e:  # 主にバリデーションエラー
        LOGGER.error(f"Error processing rhythm library: {e}")
        # バリデーションエラーの詳細はload_rhythm_library内でログ出力済み
        sys.exit(1)
    except Exception as e:
        LOGGER.error(f"An unexpected error occurred: {e}", exc_info=True)
        sys.exit(1)
