# utilities/config_loader.py
# ---------------------------
"""
Utility: load_main_cfg
~~~~~~~~~~~~~~~~~~~~~~
YAML 形式の main_cfg を辞書にパースして返す。
 - 相対パスを YAML ファイルの所在ディレクトリ基準で絶対化
 - 必須セクションやキーの存在をバリデート
 - デフォルト値を補完
 - 役割ディスパッチマップ（ROLE_DISPATCH）も注入
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from utilities import humanizer

# from some_chordmap_loader import load_chordmap  # 既存の関数 (もしあれば)

logger = logging.getLogger("otokotoba.config_loader")

# -- デフォルト必須キー一覧（存在しない場合は補完 or エラー） --
REQUIRED_TOP_LEVEL = {"global_settings", "paths", "part_defaults"}
REQUIRED_GLOBAL_SETTINGS = {"time_signature", "tempo_bpm", "key_tonic", "key_mode"}
REQUIRED_PATHS = {"chordmap_path", "rhythm_library_path", "output_dir"}


# --- 許容される役割リスト（ドキュメント・自動補完・静的チェック用） ---
ALLOWED_ROLES = [
    "melody",
    "counter",
    "pad",
    "riff",
    "rhythm",
    "bass",
    "unison",
    "sax",
]


# --- 役割ディスパッチマップ（GeneratorFactory と共有） ---
def _get_role_dispatch():
    """GeneratorFactory と同じマッピングを返すだけ (単一責任)"""
    from utilities.generator_factory import ROLE_DISPATCH  # 遅延 import
    return ROLE_DISPATCH


# ------------------------------------------------------------
def load_main_cfg(path: str | Path, *, strict: bool = True) -> dict[str, Any]:
    """
    Load main_cfg.yml and return an expanded dict.

    Parameters
    ----------
    path : str | Path
        Path to main_cfg.yml
    strict : bool
        True  -> 必須キーが無い場合 ValueError
        False -> 不足分は空辞書/None で補完して警告ログのみ

    Returns
    -------
    cfg : Dict[str, Any]
        全パスが絶対化され、デフォルト補完済みの辞書
    """
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f) or {}

    # 必須キー検証
    try:
        _validate(cfg)
    except ValueError as e:
        if strict:
            raise
        logger.warning(str(e) + " (strict=False → continue)")

    # 相対パス → 絶対パス
    base_dir = path.parent
    for key, p in cfg.get("paths", {}).items():
        cfg["paths"][key] = _abspath(base_dir, p)

    # tempo_map_path / groove_profile_path も絶対化
    gset = cfg.get("global_settings", {})
    for pkey in ("tempo_map_path", "groove_profile_path"):
        if pkey in gset:
            gset[pkey] = _abspath(base_dir, gset[pkey])

    # デフォルト Humanize プロファイルが無ければ空 dict で補完
    cfg.setdefault("humanize_profiles", {})

    # ------------------------------------------------------------
    # 役割ディスパッチテーブルを cfg に埋め込む
    # ------------------------------------------------------------
    # import を循環依存なしで遅延評価できるよう lambda で包む
    # 1. まず role_dispatch を生成してセット
    cfg["role_dispatch"] = _get_role_dispatch()

    # 2. その後でバリデーション
    _validate_roles(cfg)

    # 3. lambda でファクトリーも必要なら残す
    cfg["role_dispatch_factory"] = lambda: {
        "melody": _lazy_import("generator.melody_generator", "MelodyGenerator"),
        "counter": _lazy_import("generator.melody_generator", "MelodyGenerator"),
        "riff": _lazy_import("generator.melody_generator", "MelodyGenerator"),
        "pad": _lazy_import("generator.strings_generator", "StringsGenerator"),
        "unison": _lazy_import("generator.strings_generator", "StringsGenerator"),
        "rhythm": _lazy_import("generator.guitar_generator", "GuitarGenerator"),
        "bass": _lazy_import("generator.bass_generator", "BassGenerator"),
    }

    # セクション別 override 用ヘルパーをインジェクト（任意）
    cfg["get_section_cfg"] = lambda sec: _merge_section_override(cfg, sec)

    # part_defaults の role 値バリデーション
    _validate_roles(cfg)

    humanizer.load_profiles(cfg.get("humanize_profiles", {}))  # ★追加
    gset_flags = cfg.get("global_settings", {})
    humanizer.set_cc_flags(
        bool(gset_flags.get("use_expr_cc11", False)),
        bool(gset_flags.get("use_aftertouch", False)),
    )

    return cfg


def _abspath(base_dir: Path, path_or_none):
    """相対パスなら base_dir を前置して絶対パス文字列にする"""
    if isinstance(path_or_none, list):
        return [str((base_dir / Path(p)).expanduser().resolve()) for p in path_or_none]
    if not path_or_none:
        return ""
    p = Path(path_or_none).expanduser()
    if not p.is_absolute():
        p = base_dir / p
    return str(p.resolve())


def _validate(cfg: Mapping[str, Any]) -> None:
    """シンプルな必須キー検証。エラーなら例外を投げる。"""
    missing = REQUIRED_TOP_LEVEL - cfg.keys()
    if missing:
        raise ValueError(f"main_cfg.yml: missing section(s): {sorted(missing)}")
    g = cfg["global_settings"]
    pg = REQUIRED_GLOBAL_SETTINGS - g.keys()
    if pg:
        raise ValueError(f"[global_settings] missing key(s): {sorted(pg)}")
    p = cfg["paths"]
    pp = REQUIRED_PATHS - p.keys()
    if pp:
        raise ValueError(f"[paths] missing key(s): {sorted(pp)}")


def _validate_roles(cfg: dict[str, Any]) -> None:
    """part_defaults の role 値が role_dispatch に存在するかチェック"""
    role_dispatch = cfg.get("role_dispatch", {})
    valid_roles = set(role_dispatch.keys())
    # ALLOWED_ROLES も警告メッセージに明示
    for part, part_cfg in cfg.get("part_defaults", {}).items():
        role = part_cfg.get("role", "melody")
        if role not in valid_roles:
            logger.warning(
                f"[config_loader] part_defaults[{part}].role='{role}' は未定義の役割です。"
                f"許容値: {sorted(valid_roles)} / ドキュメント: {ALLOWED_ROLES}"
            )


# ------------------------------------------------------------
# internal helper
# ------------------------------------------------------------
def _lazy_import(module_path: str, cls_name: str):
    """Import on first access to avoid circular deps"""

    def _factory(*args, **kwargs):
        module = __import__(module_path, fromlist=[cls_name])
        return getattr(module, cls_name)(*args, **kwargs)

    return _factory


def _merge_section_override(cfg: dict[str, Any], section_name: str) -> dict[str, Any]:
    """
    元の cfg から section_overrides を取り出し、対象セクションへの
    上書きをマージした辞書を返す（deep merge は最小限）。
    """
    override = cfg.get("section_overrides", {}).get(section_name, {})
    merged = {**cfg["global_settings"], **override}  # 後勝ち
    return merged


# 既存の load_chordmap を load_chordmap_yaml という名前でも呼べるようエイリアス
# (もし load_chordmap がこのファイル内にあれば、それを直接使うか、
#  別ファイルにあれば from .some_module import load_chordmap のように読み込む)

# 例: このファイル内に load_chordmap があると仮定
# def load_chordmap(path: Path, ...):
#     # ... 既存のChordMap読み込み処理 ...
#     pass


def load_chordmap_yaml(path: Path | str) -> Any:  # ChordMapの型に合わせて修正
    """
    YAML 形式の ChordMap を読み込む。(実際の処理は既存関数を呼び出すか、ここに実装)
    """
    # もし既存の load_chordmap があれば:
    # return load_chordmap(path)

    # ここに直接YAMLを読み込む処理を実装する場合:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Chordmap file not found: {path}")
    with open(path, encoding="utf-8") as f:
        # ここでは単純にyaml.safe_loadを呼ぶ例。
        # 実際にはChordMapオブジェクトを返すなど、適切な処理が必要。
        data = yaml.safe_load(f)
    # 必要であればここでChordMapオブジェクトに変換する処理などを追加
    # from data_models.chordmap import ChordMap  # 例
    # return ChordMap(**data) # Pydanticモデルなどを使っている場合
    return data  # とりあえずパースした辞書を返す例
