"""
generator package -- 各楽器パートの音楽生成ロジック（ジェネレータ）を提供
--------------------------------------------------------------------------
このパッケージは、楽曲を構成する主要な楽器パート（ピアノ、ギター、ベース、
メロディ、ドラム、ボーカル）を生成するためのクラス群を公開します。
また、和音のボイシングを担当する ChordVoicer も提供します。

各ジェネレータは、楽曲全体の構造情報 (processed_chord_stream) と、
自身に割り当てられたパラメータ (part_params) を受け取り、
music21 の Part オブジェクトを生成して返します。

公開クラス:
    - PianoGenerator
    - GuitarGenerator
    - BassGenerator
    - MelodyGenerator
    - DrumGenerator
    - SaxGenerator
    - VocalGenerator
    - ChordVoicer
"""

# --- 各楽器ジェネレータクラスとChordVoicerを再エクスポート ---
from .base_part_generator import BasePartGenerator
from .bass_generator import BassGenerator
from .chord_voicer import ChordVoicer
# Heavy generators rely on optional data science deps (pandas, etc.).
# Import them lazily so lightweight unit tests can run without those extras.
try:  # pragma: no cover - optional dependency shims
    from .drum_generator import DrumGenerator
except ModuleNotFoundError as exc:  # pragma: no cover
    class DrumGenerator:  # type: ignore
        """Placeholder that signals missing optional dependencies."""

        def __init__(self, *_args, **_kwargs) -> None:
            raise ModuleNotFoundError(
                "DrumGenerator requires optional dependencies (e.g. pandas)."
            ) from exc
from .guitar_generator import GuitarGenerator  # ファイル名が guitar_generator.py であることを確認
from .melody_generator import MelodyGenerator
from .modular_composer_stub import ModularComposer
from .piano_generator import PianoGenerator
from .piano_template_generator import PianoTemplateGenerator
from .piano_transformer import PianoTransformer
from .piano_ml_generator import PianoMLGenerator
from .sax_generator import SaxGenerator
from .strings_generator import (
    StringsGenerator,
    EXEC_STYLE_TRILL,
    EXEC_STYLE_TREMOLO,
)
from .vocal_generator import VocalGenerator
try:  # pragma: no cover - optional dependency shims
    from modular_composer.perc_generator import PercGenerator
except ModuleNotFoundError as exc:  # pragma: no cover
    class PercGenerator:  # type: ignore
        """Placeholder that signals missing optional dependencies."""

        def __init__(self, *_args, **_kwargs) -> None:
            raise ModuleNotFoundError(
                "PercGenerator requires optional dependencies."
            ) from exc
from .arranger import Arranger

__all__ = [
    "BasePartGenerator",
    "PianoGenerator",
    "PianoTemplateGenerator",
    "PianoTransformer",
    "PianoMLGenerator",
    "GuitarGenerator",
    "BassGenerator",
    "MelodyGenerator",
    "DrumGenerator",
    "VocalGenerator",
    "SaxGenerator",
    "StringsGenerator",
    "EXEC_STYLE_TRILL",
    "EXEC_STYLE_TREMOLO",
    "ChordVoicer",
    "ModularComposer",
    "PercGenerator",
    "Arranger",
]
