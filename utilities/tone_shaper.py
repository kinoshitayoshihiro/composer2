from __future__ import annotations


try:
    from sklearn.neighbors import KNeighborsClassifier  # type: ignore
except Exception:  # pragma: no cover - optional
    KNeighborsClassifier = None


class ToneShaper:
    """Select amp/cabinet presets and emit CC events."""

    def __init__(self, presets: dict[str, int] | None = None) -> None:
        self.presets = {"clean": 0, "drive": 32, "svt": 64, "fuzz": 96}
        if presets:
            self.presets.update(presets)
        self._knn = None

    def choose_preset(self, avg_velocity: float, intensity: str) -> str:
        """Return preset name derived from intensity and average velocity."""
        score = 0
        if intensity in {"very_high", "high"}:
            score += 2
        elif intensity in {"medium_high", "medium"}:
            score += 1
        elif intensity in {"medium_low", "low"}:
            score -= 1
        if avg_velocity >= 90:
            score += 2
        elif avg_velocity >= 75:
            score += 1
        elif avg_velocity < 60:
            score -= 1
        if score >= 3:
            return "fuzz"
        if score == 2:
            return "drive"
        if score >= 0:
            return "svt"
        return "clean"

    def to_cc_events(self, preset_name: str, offset: float) -> list[dict]:
        value = self.presets.get(preset_name, self.presets["clean"])
        return [{"time": float(offset), "cc": 31, "val": value}]

    def fit(self, preset_samples: dict[str, "np.ndarray"]) -> None:
        """Fit KNN model from preset MFCC samples."""
        import numpy as np
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

    def predict_preset(self, mfcc: "np.ndarray") -> str:
        import warnings
        if self._knn is None or KNeighborsClassifier is None:  # pragma: no cover - optional
            warnings.warn(
                "scikit-learn not installed or ToneShaper not fitted; returning default",
                RuntimeWarning,
            )
            return "clean"
        import numpy as np

        feat = np.asarray(mfcc)
        if feat.ndim != 2:
            raise ValueError("MFCC array must be 2D")
        return str(self._knn.predict([feat.mean(axis=1)])[0])

__all__ = ["ToneShaper"]
