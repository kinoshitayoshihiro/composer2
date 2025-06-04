# ===============================================================
# S‑1 Sprint  ─ 共通基底クラス + BassGenerator 置換パッチ
# ===============================================================
# 1. base_part_generator.py  (NEW FILE)
# ---------------------------------------------------------------
from abc import ABC, abstractmethod
from typing import Dict, Any, List
from music21 import stream
from utilities.prettymidi_sync import apply_groove_pretty, load_groove_profile
from utilities.override_loader import get_part_override


class BasePartGenerator(ABC):
    """全楽器ジェネレーターが継承する共通基底クラス。"""

    def __init__(self, part_name: str, rhythm_lib: Dict[str, Any]):
        self.part_name = part_name
        self.rhythm_lib = rhythm_lib
        self.measure_duration = 4.0  # 4/4 基準（TODO: 動的に）
        self.logger = self._get_logger()

    # ----------------  PUBLIC API  ----------------
    def compose(
        self,
        *,
        section: Dict[str, Any],
        overrides_root: Dict[str, Any],
        groove_path: str
    ) -> stream.Part:
        """共通ワークフロー：オーバーライド適用 → ノート生成 → グルーヴ適用。"""
        self.overrides = get_part_override(
            overrides_root, section["label"], self.part_name
        )
        part = self._render_part(section)
        # groove 適用（存在すれば）
        if groove_path and part.notes:
            gp = load_groove_profile(groove_path)
            apply_groove_pretty(part, gp)
        return part

    # ----------------  必須サブクラス実装  ----------------
    @abstractmethod
    def _render_part(self, section: Dict[str, Any]) -> stream.Part:
        """各楽器固有のノート生成を行う抽象メソッド。"""
        raise NotImplementedError

    # ----------------  共通ユーティリティ  ----------------
    def _get_logger(self):
        import logging

        return logging.getLogger(self.part_name)

    def humanize_velocity(self, note_obj, ratio=0.05):
        import random

        delta = int(note_obj.volume.velocity * ratio * (random.random() * 2 - 1))
        note_obj.volume.velocity = max(1, min(127, note_obj.volume.velocity + delta))


# ===============================================================
# 2. bass_generator.py  (MODIFIED) — クラス定義を Base 継承に変更
# ---------------------------------------------------------------
from base_part_generator import BasePartGenerator
from music21 import stream, note, harmony

...


class BassGenerator(BasePartGenerator):
    def __init__(self, rhythm_lib):
        super().__init__("bass", rhythm_lib)
        # bass 固有設定
        self.base_velocity = 72

    # --------- メイン実装 ---------
    def _render_part(self, section: Dict[str, Any]) -> stream.Part:
        part = stream.Part(id="bass")
        pattern_key = self._choose_pattern(section)
        notes = self._render_pattern(pattern_key, section)
        notes = self._apply_overrides(notes)
        for n in notes:
            part.insert(n.offset, n)
        return part

    # 以下：既存メソッドを変更せずコピー or 呼び出し
    # _choose_pattern, _render_pattern, _apply_overrides など…


# ===============================================================
# 3. modular_composer.py — 生成器初期化＆呼び出し変更
# ---------------------------------------------------------------
from base_part_generator import BasePartGenerator  # noqa: F401 (import for side‑effect)

...
# 生成器作成
bass_gen = BassGenerator(rhythm_lib=bass_patterns)
...
for sec in sections:
    bass_part = bass_gen.compose(
        section=sec,
        overrides_root=arrangement_overrides,
        groove_path=args.groove_profile,
    )
    score.insert(sec["absolute_offset"], bass_part)

# ===============================================================
# 4. utilities/override_loader.py — get_part_override は既存
# ===============================================================

# === 完了 ===
