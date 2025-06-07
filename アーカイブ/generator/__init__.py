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
    - VocalGenerator
    - ChordVoicer
"""

# --- 各楽器ジェネレータクラスとChordVoicerを再エクスポート ---
from .base_part_generator import BasePartGenerator
from .piano_generator import PianoGenerator
from .guitar_generator import GuitarGenerator   # ファイル名が guitar_generator.py であることを確認
from .bass_generator import BassGenerator
from .melody_generator import MelodyGenerator
from .drum_generator import DrumGenerator
from .vocal_generator import VocalGenerator
from .chord_voicer import ChordVoicer

__all__ = [
    "BasePartGenerator",
    "PianoGenerator",
    "GuitarGenerator",
    "BassGenerator",
    "MelodyGenerator",
    "DrumGenerator",
    "VocalGenerator",
    "ChordVoicer",
]