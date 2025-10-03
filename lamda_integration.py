"""
Enhanced Arranger with LAMDa Integration
既存のArrangerクラスにLAMDaデータセットベースの推薦機能を追加
"""

import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional
import random
import logging

logger = logging.getLogger(__name__)


class LAMDaIntegratedArranger:
    """LAMDaデータセットを統合したアレンジャー"""

    def __init__(self, lamda_db_path: Optional[Path] = None):
        self.lamda_db_path = lamda_db_path or Path("data/lamda_progressions.db")
        self.use_lamda = self.lamda_db_path.exists()

        if self.use_lamda:
            logger.info("LAMDa database found - enabling data-driven progressions")
        else:
            logger.info("LAMDa database not found - using template progressions only")

    def _select_progression_enhanced(
        self, section_name: str, emotion: str, meta: Dict[str, Any]
    ) -> List[str]:
        """
        拡張されたコード進行選択
        1. メタデータ指定がある場合はそれを使用
        2. LAMDaデータベースから推薦
        3. フォールバック: テンプレート使用
        """

        # 1. 明示的な指定がある場合
        if "progression" in meta:
            return list(meta["progression"])

        # 2. LAMDaデータベースから推薦
        if self.use_lamda:
            lamda_progression = self._get_lamda_recommendation(emotion, meta)
            if lamda_progression:
                logger.info(f"Using LAMDa recommendation for {emotion}: {lamda_progression}")
                return lamda_progression

        # 3. フォールバック: 既存のテンプレート
        return self._get_template_progression(emotion)

    def _get_lamda_recommendation(self, emotion: str, meta: Dict[str, Any]) -> Optional[List[str]]:
        """LAMDaデータベースからコード進行を推薦"""
        try:
            conn = sqlite3.connect(self.lamda_db_path)
            cursor = conn.cursor()

            # メタデータからパラメータ抽出
            desired_key = meta.get("key", None)
            complexity = meta.get("complexity", 0.5)
            min_length = meta.get("min_chords", 4)
            max_length = meta.get("max_chords", 8)

            # 感情→音楽特性マッピング
            emotion_mapping = {
                "happy": {"quality": "major", "complexity_boost": 0.1},
                "sad": {"quality": "minor", "complexity_boost": -0.1},
                "energetic": {"quality": "major", "complexity_boost": 0.2},
                "calm": {"quality": "minor", "complexity_boost": -0.2},
                "dramatic": {"quality": "minor", "complexity_boost": 0.3},
                "nostalgic": {"quality": "minor", "complexity_boost": 0.0},
                "uplifting": {"quality": "major", "complexity_boost": 0.1},
            }

            emotion_params = emotion_mapping.get(
                emotion.lower(), {"quality": "major", "complexity_boost": 0.0}
            )

            target_quality = emotion_params["quality"]
            adjusted_complexity = max(
                0.1, min(0.9, complexity + emotion_params["complexity_boost"])
            )

            # データベースクエリ構築
            query = """
                SELECT chords, complexity_score, key_signature
                FROM progressions 
                WHERE quality = ?
                AND chord_count BETWEEN ? AND ?
                AND complexity_score BETWEEN ? AND ?
            """

            params = [
                target_quality,
                min_length,
                max_length,
                adjusted_complexity - 0.15,
                adjusted_complexity + 0.15,
            ]

            # キー指定がある場合
            if desired_key:
                query += " AND key_signature LIKE ?"
                params.append(f"{desired_key}%")

            # 複雑度でソートして上位候補を取得
            query += " ORDER BY ABS(complexity_score - ?) LIMIT 20"
            params.append(adjusted_complexity)

            cursor.execute(query, params)
            results = cursor.fetchall()
            conn.close()

            if results:
                # ランダムに選択して多様性を確保
                selected = random.choice(results)
                chords = selected[0].split("|")
                logger.info(
                    f"LAMDa recommendation: {chords} (complexity: {selected[1]:.2f}, key: {selected[2]})"
                )
                return chords

        except Exception as e:
            logger.error(f"Error getting LAMDa recommendation: {e}")

        return None

    def _get_template_progression(self, emotion: str) -> List[str]:
        """従来のテンプレートベース選択（フォールバック）"""
        # 既存のchordmapから選択（simplified version）
        templates = {
            "happy": ["C", "G", "Am", "F"],
            "sad": ["Am", "F", "C", "G"],
            "energetic": ["C", "F", "G", "C"],
            "calm": ["Am", "Dm", "G", "C"],
            "dramatic": ["Dm", "Am", "Bb", "F"],
            "nostalgic": ["Am", "F", "G", "Em"],
            "uplifting": ["C", "Am", "F", "G"],
        }

        progression = templates.get(emotion.lower(), ["C", "G", "Am", "F"])
        logger.info(f"Using template progression for {emotion}: {progression}")
        return progression

    def get_lamda_statistics(self) -> Dict[str, Any]:
        """LAMDaデータベースの統計情報を取得"""
        if not self.use_lamda:
            return {"status": "LAMDa database not available"}

        try:
            conn = sqlite3.connect(self.lamda_db_path)
            cursor = conn.cursor()

            stats = {}

            # 基本統計
            cursor.execute("SELECT COUNT(*) FROM progressions")
            stats["total_progressions"] = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(DISTINCT key_signature) FROM progressions")
            stats["unique_keys"] = cursor.fetchone()[0]

            cursor.execute("SELECT AVG(complexity_score) FROM progressions")
            stats["avg_complexity"] = round(cursor.fetchone()[0], 3)

            # 感情別統計
            cursor.execute(
                """
                SELECT quality, COUNT(*) 
                FROM progressions 
                GROUP BY quality 
                ORDER BY COUNT(*) DESC
            """
            )
            stats["quality_distribution"] = dict(cursor.fetchall())

            # 長さ分布
            cursor.execute(
                """
                SELECT chord_count, COUNT(*) 
                FROM progressions 
                GROUP BY chord_count 
                ORDER BY chord_count
            """
            )
            stats["length_distribution"] = dict(cursor.fetchall())

            conn.close()
            return stats

        except Exception as e:
            return {"error": str(e)}

    def recommend_similar_progressions(
        self, reference_chords: List[str], limit: int = 5
    ) -> List[Dict[str, Any]]:
        """与えられたコード進行に類似したものを推薦"""
        if not self.use_lamda:
            return []

        try:
            conn = sqlite3.connect(self.lamda_db_path)
            cursor = conn.cursor()

            reference_length = len(reference_chords)

            # 類似長のコード進行を検索
            cursor.execute(
                """
                SELECT hash_id, chords, key_signature, complexity_score
                FROM progressions 
                WHERE chord_count BETWEEN ? AND ?
                ORDER BY RANDOM()
                LIMIT ?
            """,
                (max(1, reference_length - 2), reference_length + 2, limit * 3),
            )

            candidates = cursor.fetchall()
            conn.close()

            # 簡易類似度計算
            similar = []
            for hash_id, chords_str, key_sig, complexity in candidates:
                chords = chords_str.split("|")
                similarity = self._calculate_similarity(reference_chords, chords)

                similar.append(
                    {
                        "hash_id": hash_id,
                        "chords": chords,
                        "key_signature": key_sig,
                        "complexity_score": complexity,
                        "similarity": similarity,
                    }
                )

            # 類似度でソート
            similar.sort(key=lambda x: x["similarity"], reverse=True)
            return similar[:limit]

        except Exception as e:
            logger.error(f"Error finding similar progressions: {e}")
            return []

    def _calculate_similarity(self, chords1: List[str], chords2: List[str]) -> float:
        """簡易的なコード進行類似度計算"""
        if not chords1 or not chords2:
            return 0.0

        # 共通コード数ベースの類似度
        set1 = set(chords1)
        set2 = set(chords2)

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        if union == 0:
            return 0.0

        jaccard = intersection / union

        # 長さ類似度も考慮
        length_similarity = 1.0 - abs(len(chords1) - len(chords2)) / max(len(chords1), len(chords2))

        return (jaccard * 0.7) + (length_similarity * 0.3)


# 使用例とテスト
if __name__ == "__main__":
    arranger = LAMDaIntegratedArranger()

    # 統計情報表示
    stats = arranger.get_lamda_statistics()
    print("LAMDa Database Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # コード進行推薦テスト
    test_emotions = ["happy", "sad", "energetic", "calm", "dramatic"]

    print("\nProgression Recommendations:")
    for emotion in test_emotions:
        progression = arranger._get_lamda_recommendation(emotion, {})
        if progression:
            print(f"  {emotion}: {' - '.join(progression)}")
        else:
            fallback = arranger._get_template_progression(emotion)
            print(f"  {emotion}: {' - '.join(fallback)} (template)")

    # 類似コード進行検索テスト
    reference = ["C", "G", "Am", "F"]
    similar = arranger.recommend_similar_progressions(reference)
    print(f"\nSimilar to {' - '.join(reference)}:")
    for item in similar:
        print(f"  {' - '.join(item['chords'])} (similarity: {item['similarity']:.2f})")
