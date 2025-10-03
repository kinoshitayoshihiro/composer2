"""
LAMDa Learning System
ユーザーの使用パターンから学習してコード進行推薦を改善
"""

import sqlite3
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)


class ProgressionLearner:
    """コード進行使用パターンの学習システム"""

    def __init__(self, usage_db_path: Path):
        self.usage_db_path = usage_db_path
        self._init_usage_database()

    def _init_usage_database(self) -> None:
        """使用履歴データベース初期化"""
        conn = sqlite3.connect(self.usage_db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS usage_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                emotion TEXT,
                section_name TEXT,
                progression TEXT,
                source TEXT,  -- 'lamda', 'template', 'custom'
                user_rating INTEGER,  -- 1-5 scale
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT  -- JSON format additional info
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS progression_preferences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                emotion TEXT,
                preferred_progression TEXT,
                usage_count INTEGER DEFAULT 1,
                avg_rating REAL,
                last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(emotion, preferred_progression)
            )
        """
        )

        conn.commit()
        conn.close()

    def record_usage(
        self,
        session_id: str,
        emotion: str,
        section: str,
        progression: List[str],
        source: str,
        rating: Optional[int] = None,
        metadata: Optional[Dict] = None,
    ) -> None:
        """コード進行の使用を記録"""
        conn = sqlite3.connect(self.usage_db_path)
        cursor = conn.cursor()

        progression_str = "|".join(progression)
        metadata_str = json.dumps(metadata or {})

        cursor.execute(
            """
            INSERT INTO usage_history 
            (session_id, emotion, section_name, progression, source, 
             user_rating, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (session_id, emotion, section, progression_str, source, rating, metadata_str),
        )

        # preferences テーブル更新
        cursor.execute(
            """
            INSERT INTO progression_preferences 
            (emotion, preferred_progression, usage_count, avg_rating)
            VALUES (?, ?, 1, ?)
            ON CONFLICT(emotion, preferred_progression) DO UPDATE SET
                usage_count = usage_count + 1,
                avg_rating = CASE 
                    WHEN ? IS NOT NULL THEN 
                        (avg_rating * (usage_count - 1) + ?) / usage_count
                    ELSE avg_rating
                END,
                last_used = CURRENT_TIMESTAMP
        """,
            (emotion, progression_str, rating, rating, rating),
        )

        conn.commit()
        conn.close()

        logger.info(f"Recorded usage: {emotion} -> {progression_str}")

    def get_learned_recommendations(self, emotion: str, limit: int = 5) -> List[Dict[str, Any]]:
        """学習結果に基づく推薦"""
        conn = sqlite3.connect(self.usage_db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT preferred_progression, usage_count, avg_rating, last_used
            FROM progression_preferences
            WHERE emotion = ?
            ORDER BY 
                (usage_count * 0.4 + COALESCE(avg_rating, 3) * 0.6) DESC,
                last_used DESC
            LIMIT ?
        """,
            (emotion, limit),
        )

        results = cursor.fetchall()
        conn.close()

        recommendations = []
        for prog_str, count, rating, last_used in results:
            recommendations.append(
                {
                    "progression": prog_str.split("|"),
                    "usage_count": count,
                    "avg_rating": rating,
                    "last_used": last_used,
                    "confidence": min(1.0, count / 10.0),  # 使用回数ベースの信頼度
                }
            )

        return recommendations

    def analyze_user_patterns(self) -> Dict[str, Any]:
        """ユーザーの使用パターン分析"""
        conn = sqlite3.connect(self.usage_db_path)
        cursor = conn.cursor()

        analysis = {}

        # 感情別使用頻度
        cursor.execute(
            """
            SELECT emotion, COUNT(*) as count
            FROM usage_history
            GROUP BY emotion
            ORDER BY count DESC
        """
        )
        analysis["emotion_frequency"] = dict(cursor.fetchall())

        # ソース別使用統計
        cursor.execute(
            """
            SELECT source, COUNT(*) as count, AVG(user_rating) as avg_rating
            FROM usage_history
            WHERE user_rating IS NOT NULL
            GROUP BY source
        """
        )
        analysis["source_performance"] = {
            row[0]: {"count": row[1], "avg_rating": row[2]} for row in cursor.fetchall()
        }

        # 時間別使用パターン
        cursor.execute(
            """
            SELECT strftime('%H', timestamp) as hour, COUNT(*) as count
            FROM usage_history
            GROUP BY hour
            ORDER BY hour
        """
        )
        analysis["hourly_usage"] = dict(cursor.fetchall())

        # 最も成功したコード進行
        cursor.execute(
            """
            SELECT progression, AVG(user_rating) as avg_rating, COUNT(*) as count
            FROM usage_history
            WHERE user_rating IS NOT NULL
            GROUP BY progression
            HAVING count >= 2
            ORDER BY avg_rating DESC, count DESC
            LIMIT 10
        """
        )
        analysis["top_progressions"] = [
            {"progression": row[0].split("|"), "avg_rating": row[1], "count": row[2]}
            for row in cursor.fetchall()
        ]

        conn.close()
        return analysis


class AdaptiveProgressionSelector:
    """学習機能付きコード進行選択器"""

    def __init__(self, lamda_db_path: Path, usage_db_path: Path):
        self.lamda_db_path = lamda_db_path
        self.learner = ProgressionLearner(usage_db_path)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def select_progression_adaptive(
        self, emotion: str, section: str, meta: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """適応的コード進行選択"""
        meta = meta or {}

        # 1. 学習済み推薦を優先チェック
        learned = self.learner.get_learned_recommendations(emotion, limit=3)

        if learned and learned[0]["confidence"] > 0.3:
            best_learned = learned[0]
            progression = best_learned["progression"]
            source = "learned"
            confidence = best_learned["confidence"]

            logger.info(
                f"Using learned progression for {emotion}: "
                f"{progression} (confidence: {confidence:.2f})"
            )

        # 2. LAMDaデータベースから選択
        elif self.lamda_db_path.exists():
            progression = self._get_lamda_progression(emotion, meta)
            source = "lamda"
            confidence = 0.6

        # 3. テンプレートフォールバック
        else:
            progression = self._get_template_progression(emotion)
            source = "template"
            confidence = 0.3

        # 使用記録
        self.learner.record_usage(self.session_id, emotion, section, progression, source)

        return {
            "progression": progression,
            "source": source,
            "confidence": confidence,
            "alternatives": learned[:2] if source != "learned" else [],
        }

    def _get_lamda_progression(self, emotion: str, meta: Dict[str, Any]) -> List[str]:
        """LAMDaから進行選択（簡略版）"""
        try:
            conn = sqlite3.connect(self.lamda_db_path)
            cursor = conn.cursor()

            quality = "major" if emotion in ["happy", "energetic"] else "minor"

            cursor.execute(
                """
                SELECT chords FROM progressions 
                WHERE quality = ? 
                ORDER BY RANDOM() 
                LIMIT 1
            """,
                (quality,),
            )

            result = cursor.fetchone()
            conn.close()

            if result:
                return result[0].split("|")

        except Exception as e:
            logger.error(f"Error selecting LAMDa progression: {e}")

        return self._get_template_progression(emotion)

    def _get_template_progression(self, emotion: str) -> List[str]:
        """テンプレート進行選択"""
        templates = {
            "happy": ["C", "G", "Am", "F"],
            "sad": ["Am", "F", "C", "G"],
            "energetic": ["C", "F", "G", "C"],
            "calm": ["Am", "Dm", "G", "C"],
        }
        return templates.get(emotion, ["C", "G", "Am", "F"])

    def provide_feedback(self, emotion: str, progression: List[str], rating: int) -> None:
        """フィードバック記録"""
        self.learner.record_usage(
            self.session_id, emotion, "feedback", progression, "feedback", rating
        )
        logger.info(f"Feedback recorded: {emotion} -> " f"{progression} (rating: {rating})")

    def get_user_insights(self) -> Dict[str, Any]:
        """ユーザー使用パターンの洞察"""
        return self.learner.analyze_user_patterns()


# 使用例
if __name__ == "__main__":
    lamda_db = Path("data/lamda_progressions.db")
    usage_db = Path("data/user_preferences.db")

    selector = AdaptiveProgressionSelector(lamda_db, usage_db)

    # 選択テスト
    result = selector.select_progression_adaptive("happy", "verse")
    print(f"Selected: {result['progression']} (source: {result['source']})")

    # フィードバック
    selector.provide_feedback("happy", result["progression"], 4)

    # 洞察
    insights = selector.get_user_insights()
    print(f"User patterns: {insights}")
