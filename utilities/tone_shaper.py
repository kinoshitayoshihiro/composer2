from __future__ import annotations
from typing import Dict, List
import numpy as np
from numpy.typing import NDArray

try:
    from sklearn.neighbors import KNeighborsClassifier  # type: ignore
except Exception:  # pragma: no cover - optional
    KNeighborsClassifier = None

# ──────────────────────────────────────────────────────────
# プリセット表 (Intensity × Loud/Soft)
# ──────────────────────────────────────────────────────────
PRESET_TABLE = {
    ("low",    "soft"):  "clean",
    ("low",    "loud"):  "crunch",
    ("medium", "soft"):  "crunch",
    ("medium", "loud"):  "drive",
    ("high",   "soft"):  "drive",
    ("high",   "loud"):  "fuzz",
}


class ToneShaper:
    """Select amp/cabinet presets and emit CC events."""

    def __init__(self, presets: dict[str, int] | None = None) -> None:
        # CC31 値マップ
        self.presets: Dict[str, int] = {"clean": 0, "crunch": 32, "drive": 64, "fuzz": 96}
        if presets:
            self.presets.update(presets)
        self._knn = None

    # ──────────────────────────────────────────────
    # 競合解消後の choose_preset
    # ──────────────────────────────────────────────
    def choose_preset(self, avg_velocity: float, intensity: str) -> str:
        """
        強度 (intensity) と 平均ベロシティからプリセット名を返す。

        1) テーブル方式 (PRESET_TABLE) を優先
        2) テーブルに無ければ簡易スコア方式でフォールバック
        """
        # 1) テーブル方式 -------------
        vel_bucket = "loud" if avg_velocity >= 60 else "soft"
        int_bucket = (
            "high"   if intensity in {"very_high", "high"} else
            "medium" if intensity in {"medium_high", "medium"} else
            "low"
        )
        preset = PRESET_TABLE.get((int_bucket, vel_bucket))
        if preset:
            return preset

        # 2) フォールバックスコア方式 ----
        score = 0
        score += {"very_high": 2, "high": 2,
                  "medium_high": 1, "medium": 1,
                  "medium_low": -1, "low": -1}.get(intensity, 0)

        if avg_velocity >= 90:
            score += 2
        elif avg_velocity >= 75:
            score += 1
        elif avg_velocity < 60:
            score -= 1

        if score >= 5:
            return "fuzz"
        if score >= 3:
            return "drive"
        if score >= 0:
            return "crunch"
        return "clean"

    # ──────────────────────────────────────────────
    def to_cc_events(self, preset_name: str, offset: float) -> List[dict]:
        value = self.presets.get(preset_name, self.presets["clean"])
        return [{"time": float(offset), "cc": 31, "val": value}]

    def fit(self, preset_samples: dict[str, NDArray[np.floating]]) -> None:
        """Fit KNN model from preset MFCC samples."""
        if KNeighborsClassifier is None:  # pragma: no cover - optional
            import warnings
            warnings.warn(
                "scikit-learn not installed; ToneShaper disabled", RuntimeWarning
            )
            return
        X = []
        y = []
        for name, mfcc in preset_samples.items():
            arr = np.asarray(mfcc)
            if arr.ndim != 2:
                raise ValueError("MFCC array must be 2D")
            X.append(arr.mean(axis=1))
            y.append(name)
        self._knn = KNeighborsClassifier(n_neighbors=1)
        self._knn.fit(X, y)

    def predict_preset(self, mfcc: NDArray[np.floating]) -> str:
        import warnings
        if self._knn is None or KNeighborsClassifier is None:  # pragma: no cover - optional
            warnings.warn(
                "scikit-learn not installed or ToneShaper not fitted; returning default",
                RuntimeWarning,
            )
            return "clean"
        feat = np.asarray(mfcc)
        if feat.ndim != 2:
            raise ValueError("MFCC array must be 2D")
        return str(self._knn.predict([feat.mean(axis=1)])[0])

__all__ = ["ToneShaper"]