from __future__ import annotations


try:
    from sklearn.neighbors import KNeighborsClassifier  # type: ignore
except Exception:  # pragma: no cover - optional
    KNeighborsClassifier = None


class ToneShaper:
    """Select amp/cabinet presets and emit CC events."""

    def __init__(
        self,
        preset_map: dict[str, dict[str, int] | int] | None = None,
        ir_map: dict[str, str] | None = None,
        default_preset: str = "clean",
        rules: list[dict[str, str]] | None = None,
    ) -> None:
        self.preset_map: dict[str, dict[str, int]] = {"clean": {"amp": 20}}
        if preset_map:
            for name, data in preset_map.items():
                if isinstance(data, int):
                    self.preset_map[name] = {"amp": int(data)}
                else:
                    self.preset_map[name] = {k: int(v) for k, v in data.items()}
        self.ir_map = ir_map or {}
        self.default_preset = default_preset
        self.rules = rules or []
        self._selected = default_preset
        self._knn = None

    @classmethod
    def from_yaml(cls, path: str | "Path") -> "ToneShaper":
        import yaml
        from pathlib import Path

        path = Path(path)
        if not path.is_file():
            return cls()
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        presets = data.get("presets") or {}
        levels = data.get("levels") or {}
        ir_map = data.get("ir") or {}
        rules = data.get("rules") or []
        preset_map: dict[str, dict[str, int]] = {}
        for name, val in presets.items():
            entry = {"amp": int(val)}
            if name in levels:
                for k, v in levels[name].items():
                    entry[k] = int(v)
            preset_map[name] = entry
        return cls(preset_map=preset_map, ir_map=ir_map, rules=rules)

    def choose_preset(
        self,
        amp_preset: str | None,
        intensity: str | None,
        avg_velocity: float,
    ) -> str:
        """Return preset name based on explicit label or heuristics."""
        if amp_preset:
            chosen = amp_preset
        else:
            env = {
                "intensity": str(intensity or "medium").lower(),
                "avg_vel": avg_velocity,
                "avg_velocity": avg_velocity,
            }
            chosen = None
            for rule in self.rules:
                cond = rule.get("if")
                preset = rule.get("preset")
                if not cond or not preset:
                    continue
                try:
                    if eval(cond, {"__builtins__": {}}, env):
                        chosen = preset
                        break
                except Exception:
                    continue
            if not chosen:
                level = env["intensity"]
                if level == "high" or avg_velocity > 100:
                    chosen = "drive"
                elif level == "medium" or avg_velocity > 60:
                    chosen = "crunch"
                else:
                    chosen = "clean"
        if chosen not in self.preset_map:
            chosen = self.default_preset
        self._selected = chosen
        return chosen

    def to_cc_events(
        self,
        offset_ql: float = 0.0,
        cc_amp: int = 31,
        cc_rev: int = 91,
        cc_cho: int = 93,
        cc_del: int = 94,
        *,
        as_dict: bool = False,
    ) -> list[tuple[float, int, int] | dict[str, int | float]]:
        """Return CC events for the last chosen preset."""
        preset = self.preset_map.get(self._selected) or self.preset_map.get(self.default_preset, {})
        amp = max(0, min(127, int(preset.get("amp", 0))))
        base = amp
        rev = max(0, min(127, int(preset.get("reverb", int(base * 0.3)))))
        cho = max(0, min(127, int(preset.get("chorus", int(base * 0.3)))))
        dly = max(0, min(127, int(preset.get("delay", int(base * 0.3)))))
        events: list[tuple[float, int, int]] = [
            (float(offset_ql), int(cc_amp), amp),
            (float(offset_ql), int(cc_rev), rev),
            (float(offset_ql), int(cc_cho), cho),
            (float(offset_ql), int(cc_del), dly),
        ]
        if as_dict:
            return [{"time": o, "cc": c, "val": v} for o, c, v in events]
        return events

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
