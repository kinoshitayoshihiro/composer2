"""
LAMDa Dataset Analyzer for Chord Progression Mining
音楽的パターンの抽出とデータベース構築を行います
"""

import pickle
import sqlite3
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Any
import music21
from music21 import chord, key, roman, stream
import numpy as np
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ChordProgression:
    """コード進行の分析結果"""

    hash_id: str
    chords: List[str]
    key_signature: str
    quality: str  # major/minor
    harmonic_function: List[str]  # I, V, vi, etc.
    tempo_estimate: float
    complexity_score: float


class LAMDaAnalyzer:
    """LAMDaデータセットからコード進行パターンを抽出"""

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.db_path = self.data_dir.parent / "lamda_progressions.db"
        self.progressions: List[ChordProgression] = []

    def analyze_all_files(self) -> None:
        """全pickleファイルを分析してデータベースに保存"""
        pickle_files = list(self.data_dir.glob("*.pickle"))
        logger.info(f"Found {len(pickle_files)} pickle files to analyze")

        self._init_database()

        for i, file_path in enumerate(pickle_files):
            logger.info(f"Processing {file_path.name} ({i+1}/{len(pickle_files)})")
            self._analyze_file(file_path)

            # バッチごとにDBに保存
            if (i + 1) % 10 == 0:
                self._save_to_database()
                self.progressions.clear()

        # 残りを保存
        if self.progressions:
            self._save_to_database()

    def _analyze_file(self, file_path: Path) -> None:
        """単一pickleファイルを分析"""
        try:
            with open(file_path, "rb") as f:
                samples = pickle.load(f)

            for hash_id, midi_events in samples:
                progression = self._extract_chord_progression(hash_id, midi_events)
                if progression:
                    self.progressions.append(progression)

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")

    def _extract_chord_progression(self, hash_id: str, midi_events: List) -> ChordProgression:
        """MIDIイベントからコード進行を抽出"""
        try:
            # MIDIイベントをmusic21 Streamに変換
            s = stream.Stream()

            # イベントをノートに変換
            for event in midi_events:
                if len(event) >= 3:
                    time, pitch, velocity = event[0], event[1], event[2]
                    if velocity > 0:  # Note on
                        note = music21.note.Note(pitch, quarterLength=0.5)
                        s.insert(time * 0.125, note)  # タイミングスケール調整

            if len(s.notes) == 0:
                return None

            # コード分析
            chords = []
            key_sig = None

            # 4拍ごとにセグメント化してコード分析
            duration = s.duration.quarterLength
            segments = int(duration // 1) + 1

            for i in range(segments):
                start = i * 1.0
                end = (i + 1) * 1.0
                segment = s.getElementsByOffset(start, end)

                if segment.notes:
                    pitches = [n.pitch for n in segment.notes]
                    if len(pitches) >= 2:  # 最低2音以上
                        try:
                            detected_chord = chord.Chord(pitches).simplifyEnharmonics()
                            chord_symbol = detected_chord.figure
                            if chord_symbol:
                                chords.append(chord_symbol)
                        except:
                            continue

            if not chords:
                return None

            # キー推定
            try:
                key_sig = s.analyze("key")
                key_name = f"{key_sig.tonic.name} {key_sig.mode}"
                quality = key_sig.mode
            except:
                key_name = "C major"
                quality = "major"

            # ローマ数字分析
            harmonic_functions = []
            try:
                if key_sig:
                    for chord_symbol in chords:
                        c = chord.Chord(chord_symbol)
                        rn = roman.romanNumeralFromChord(c, key_sig)
                        harmonic_functions.append(str(rn.figure))
            except:
                harmonic_functions = chords.copy()

            # 複雑度スコア計算
            complexity = self._calculate_complexity(chords, harmonic_functions)

            # テンポ推定（簡易版）
            tempo_estimate = 120.0  # デフォルト

            return ChordProgression(
                hash_id=hash_id,
                chords=chords,
                key_signature=key_name,
                quality=quality,
                harmonic_function=harmonic_functions,
                tempo_estimate=tempo_estimate,
                complexity_score=complexity,
            )

        except Exception as e:
            logger.debug(f"Failed to extract progression from {hash_id}: {e}")
            return None

    def _calculate_complexity(self, chords: List[str], functions: List[str]) -> float:
        """コード進行の複雑度を計算"""
        if not chords:
            return 0.0

        # ユニークコード数
        unique_chords = len(set(chords))
        # 進行の長さ
        length = len(chords)
        # 転調回数（簡易推定）
        modulations = 0

        # 複雑度スコア = (ユニーク数 / 長さ) + 転調ボーナス
        complexity = (unique_chords / length) + (modulations * 0.2)
        return min(complexity, 1.0)

    def _init_database(self) -> None:
        """SQLiteデータベースを初期化"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS progressions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                hash_id TEXT UNIQUE,
                chords TEXT,
                key_signature TEXT,
                quality TEXT,
                harmonic_function TEXT,
                tempo_estimate REAL,
                complexity_score REAL,
                chord_count INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # インデックス作成
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_quality ON progressions(quality)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_key ON progressions(key_signature)")
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_complexity ON progressions(complexity_score)"
        )

        conn.commit()
        conn.close()

    def _save_to_database(self) -> None:
        """分析結果をデータベースに保存"""
        if not self.progressions:
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for prog in self.progressions:
            try:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO progressions 
                    (hash_id, chords, key_signature, quality, harmonic_function, 
                     tempo_estimate, complexity_score, chord_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        prog.hash_id,
                        "|".join(prog.chords),
                        prog.key_signature,
                        prog.quality,
                        "|".join(prog.harmonic_function),
                        prog.tempo_estimate,
                        prog.complexity_score,
                        len(prog.chords),
                    ),
                )
            except sqlite3.IntegrityError:
                continue  # 重複スキップ

        conn.commit()
        conn.close()
        logger.info(f"Saved {len(self.progressions)} progressions to database")


class ProgressionRecommender:
    """コード進行推薦システム"""

    def __init__(self, db_path: Path):
        self.db_path = db_path

    def recommend_by_emotion(
        self, emotion: str, key: str = None, complexity: float = 0.5, limit: int = 10
    ) -> List[Tuple[str, List[str]]]:
        """感情とキーに基づいてコード進行を推薦"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 感情マッピング（簡易版）
        emotion_quality = {
            "happy": "major",
            "sad": "minor",
            "energetic": "major",
            "calm": "minor",
            "dramatic": "minor",
        }.get(emotion.lower(), "major")

        query = """
            SELECT hash_id, chords, complexity_score 
            FROM progressions 
            WHERE quality = ? 
            AND complexity_score BETWEEN ? AND ?
        """
        params = [emotion_quality, complexity - 0.2, complexity + 0.2]

        if key:
            query += " AND key_signature LIKE ?"
            params.append(f"{key}%")

        query += " ORDER BY ABS(complexity_score - ?) LIMIT ?"
        params.extend([complexity, limit])

        cursor.execute(query, params)
        results = cursor.fetchall()
        conn.close()

        return [(row[0], row[1].split("|")) for row in results]

    def get_statistics(self) -> Dict[str, Any]:
        """データベース統計を取得"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        stats = {}

        # 総数
        cursor.execute("SELECT COUNT(*) FROM progressions")
        stats["total_progressions"] = cursor.fetchone()[0]

        # キー分布
        cursor.execute(
            "SELECT key_signature, COUNT(*) FROM progressions GROUP BY key_signature ORDER BY COUNT(*) DESC LIMIT 10"
        )
        stats["key_distribution"] = dict(cursor.fetchall())

        # 長さ分布
        cursor.execute(
            "SELECT chord_count, COUNT(*) FROM progressions GROUP BY chord_count ORDER BY chord_count"
        )
        stats["length_distribution"] = dict(cursor.fetchall())

        # 複雑度分布
        cursor.execute(
            "SELECT ROUND(complexity_score, 1) as complexity, COUNT(*) FROM progressions GROUP BY ROUND(complexity_score, 1) ORDER BY complexity"
        )
        stats["complexity_distribution"] = dict(cursor.fetchall())

        conn.close()
        return stats


if __name__ == "__main__":
    # 使用例
    data_dir = Path("data/Los-Angeles/CHORDS_DATA")
    analyzer = LAMDaAnalyzer(data_dir)

    print("LAMDaデータセット分析を開始...")
    analyzer.analyze_all_files()

    # 推薦システムのテスト
    recommender = ProgressionRecommender(analyzer.db_path)
    stats = recommender.get_statistics()
    print(f"\n分析完了: {stats}")

    # 推薦例
    recommendations = recommender.recommend_by_emotion("happy", "C", complexity=0.3)
    print(f"\nハッピーな感情のコード進行推薦:")
    for hash_id, chords in recommendations[:5]:
        print(f"  {' - '.join(chords)}")
