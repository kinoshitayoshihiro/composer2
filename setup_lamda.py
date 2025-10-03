#!/usr/bin/env python3
"""
LAMDa Dataset Utilization Setup
LAMDaデータセットを活用するためのセットアップスクリプト
"""

import sys
from pathlib import Path
import argparse
import logging
from typing import Optional
import time

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent))

from lamda_analyzer import LAMDaAnalyzer, ProgressionRecommender
from adaptive_learning import AdaptiveProgressionSelector

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup_lamda_system(data_dir: Path, force_reanalysis: bool = False) -> bool:
    """LAMDaシステム全体のセットアップ"""

    logger.info("=" * 60)
    logger.info("LAMDa Dataset Utilization System Setup")
    logger.info("=" * 60)

    # 1. データディレクトリ確認
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return False

    pickle_files = list(data_dir.glob("*.pickle"))
    if not pickle_files:
        logger.error(f"No pickle files found in {data_dir}")
        return False

    logger.info(f"Found {len(pickle_files)} pickle files to process")

    # 2. データベース構築
    db_path = data_dir.parent / "lamda_progressions.db"

    if db_path.exists() and not force_reanalysis:
        logger.info(f"Database already exists: {db_path}")
        logger.info("Use --force to reanalyze")
    else:
        logger.info("Starting LAMDa dataset analysis...")
        start_time = time.time()

        analyzer = LAMDaAnalyzer(data_dir)
        try:
            analyzer.analyze_all_files()
            elapsed = time.time() - start_time
            logger.info(f"Analysis completed in {elapsed:.1f} seconds")
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return False

    # 3. システム検証
    logger.info("Verifying system components...")

    try:
        # データベース統計
        recommender = ProgressionRecommender(db_path)
        stats = recommender.get_statistics()

        logger.info("Database Statistics:")
        logger.info(f"  Total progressions: {stats.get('total_progressions', 0)}")
        logger.info(f"  Unique keys: {len(stats.get('key_distribution', {}))}")
        logger.info(
            f"  Length range: {min(stats.get('length_distribution', {}).keys(), default=0)}-{max(stats.get('length_distribution', {}).keys(), default=0)}"
        )

        # 推薦テスト
        test_emotions = ["happy", "sad", "energetic", "calm"]
        logger.info("\nTesting recommendations:")

        for emotion in test_emotions:
            recommendations = recommender.recommend_by_emotion(emotion, limit=2)
            if recommendations:
                hash_id, chords = recommendations[0]
                logger.info(f"  {emotion}: {' - '.join(chords)}")
            else:
                logger.warning(f"  {emotion}: No recommendations found")

        # 適応学習システムテスト
        usage_db = data_dir.parent / "user_preferences.db"
        selector = AdaptiveProgressionSelector(db_path, usage_db)

        test_result = selector.select_progression_adaptive("happy", "test")
        logger.info(
            f"\nAdaptive selector test: {test_result['progression']} "
            f"(source: {test_result['source']})"
        )

    except Exception as e:
        logger.error(f"System verification failed: {e}")
        return False

    logger.info("\n" + "=" * 60)
    logger.info("Setup completed successfully!")
    logger.info("LAMDa system is ready for integration")
    logger.info("=" * 60)

    return True


def create_integration_example() -> None:
    """統合例のコード生成"""

    example_code = '''
"""
Enhanced Arranger with LAMDa Integration Example
既存のArrangerクラスへの統合例
"""

from pathlib import Path
from lamda_integration import LAMDaIntegratedArranger
from adaptive_learning import AdaptiveProgressionSelector

class EnhancedArranger(LAMDaIntegratedArranger):
    """LAMDa統合済みArranger"""
    
    def __init__(self):
        # LAMDaデータベースパス設定
        lamda_db = Path("data/lamda_progressions.db")
        usage_db = Path("data/user_preferences.db")
        
        super().__init__(lamda_db)
        self.adaptive_selector = AdaptiveProgressionSelector(lamda_db, usage_db)
    
    def _select_progression(self, section_name: str, emotion: str, 
                          meta: dict) -> list[str]:
        """オーバーライド: 適応学習を使用"""
        
        # 適応学習による選択
        result = self.adaptive_selector.select_progression_adaptive(
            emotion, section_name, meta
        )
        
        progression = result['progression']
        confidence = result['confidence']
        
        print(f"Selected progression: {' - '.join(progression)} "
              f"(confidence: {confidence:.2f})")
        
        return progression
    
    def provide_user_feedback(self, emotion: str, progression: list[str], 
                            rating: int) -> None:
        """ユーザーフィードバック記録"""
        self.adaptive_selector.provide_feedback(emotion, progression, rating)

# 使用例
if __name__ == "__main__":
    arranger = EnhancedArranger()
    
    # コード進行生成
    progression = arranger._select_progression("verse", "happy", {})
    print(f"Generated: {progression}")
    
    # フィードバック（5段階評価）
    arranger.provide_user_feedback("happy", progression, 4)
    
    # 統計情報表示
    stats = arranger.get_lamda_statistics()
    print(f"LAMDa stats: {stats}")
'''

    output_path = Path("enhanced_arranger_example.py")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(example_code)

    logger.info(f"Integration example created: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Setup LAMDa dataset utilization system")

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/Los-Angeles-MIDI/CHORDS_DATA"),
        help="Path to LAMDa pickle files directory",
    )

    parser.add_argument(
        "--force", action="store_true", help="Force reanalysis even if database exists"
    )

    parser.add_argument(
        "--create-example", action="store_true", help="Create integration example code"
    )

    parser.add_argument(
        "--stats-only", action="store_true", help="Show statistics only, skip setup"
    )

    args = parser.parse_args()

    if args.create_example:
        create_integration_example()
        return

    if args.stats_only:
        db_path = args.data_dir.parent / "lamda_progressions.db"
        if db_path.exists():
            recommender = ProgressionRecommender(db_path)
            stats = recommender.get_statistics()
            print("Current LAMDa Database Statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
        else:
            print("Database not found. Run setup first.")
        return

    # メインセットアップ実行
    success = setup_lamda_system(args.data_dir, args.force)

    if success:
        print("\nNext steps:")
        print("1. Import LAMDaIntegratedArranger in your existing code")
        print("2. Replace _select_progression method calls")
        print("3. Use --create-example to see integration code")
        print("4. Collect user feedback to improve recommendations")
    else:
        print("Setup failed. Check logs for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
