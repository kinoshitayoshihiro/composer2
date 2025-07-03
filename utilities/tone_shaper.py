from __future__ import annotations


class ToneShaper:
    """Select amp/cabinet presets and emit CC events."""

    def __init__(self, presets: dict[str, int] | None = None) -> None:
        self.presets = {"clean": 0, "drive": 32, "svt": 64, "fuzz": 96}
        if presets:
            self.presets.update(presets)
        self._knn = None

    def choose_preset(self, avg_velocity: float, intensity: str) -> str:
        """Return preset name for given velocity and intensity."""
        if intensity in {"high", "very_high"} or avg_velocity >= 85:
            return "drive"
        if intensity in {"medium_high", "medium"}:
            return "svt"
        if intensity in {"medium_low", "low"} or avg_velocity < 60:
            return "clean"
        return "fuzz"

    def to_cc_events(self, preset_name: str, offset: float) -> list[dict]:
        value = self.presets.get(preset_name, self.presets["clean"])
        return [{"time": float(offset), "number": 31, "value": value}]

    def fit(self, preset_samples: dict[str, "np.ndarray"]) -> None:
        """Fit KNN model from preset MFCC samples."""
        import numpy as np
        from sklearn.neighbors import KNeighborsClassifier

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
        if self._knn is None:
            raise RuntimeError("ToneShaper model not fitted")
        import numpy as np

        feat = np.asarray(mfcc)
        if feat.ndim != 2:
            raise ValueError("MFCC array must be 2D")
        return str(self._knn.predict([feat.mean(axis=1)])[0])

__all__ = ["ToneShaper"]
