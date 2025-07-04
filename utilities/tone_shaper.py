from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Union

import numpy as np
from numpy.typing import NDArray

try:
    from sklearn.neighbors import KNeighborsClassifier  # type: ignore
except Exception:  # pragma: no cover – optional dependency
    KNeighborsClassifier = None

# ──────────────────────────────────────────────────────────
# プリセット表 (Intensity × Loud/Soft) ― main ブランチ由来の簡易マップ
# ──────────────────────────────────────────────────────────
PRESET_TABLE: dict[tuple[str, str], str] = {
    ("low",    "soft"): "clean",
    ("low",    "loud"): "crunch",
    ("medium", "soft"): "crunch",
    ("medium", "loud"): "drive",
    ("high",   "soft"): "drive",
    ("high",   "loud"): "fuzz",
}


class ToneShaper:
    """
    Amp / Cabinet プリセットを選択し、必要な CC イベントを生成するユーティリティ。

    - **プリセットマップ**   : プリセット名 → {"amp": 0-127, "reverb": …, …}
    - **IR マップ**        : プリセット名 → Impulse Response ファイルパス
    - **ルール**           : ``{"if": "<python-expr>", "preset": "<name>"}``
    - **KNN**  (option)   : MFCC からプリセット推定
    """

    # ------------------------------------------------------
    # constructor / loader
    # ------------------------------------------------------
    def __init__(
        self,
        preset_map: dict[str, dict[str, int] | int] | None = None,
        ir_map: dict[str, str] | None = None,
        default_preset: str = "clean",
        rules: list[dict[str, str]] | None = None,
    ) -> None:
        # 内部保持は dict[str, dict[str,int]]
        self.preset_map: dict[str, dict[str, int]] = {"clean": {"amp": 20}}
        if preset_map:
            for name, data in preset_map.items():
                if isinstance(data, int):
                    self.preset_map[name] = {"amp": int(data)}
                else:
                    self.preset_map[name] = {k: int(v) for k, v in data.items()}

        self.ir_map: dict[str, str] = ir_map or {}
        self.default_preset: str = default_preset
        self.rules: list[dict[str, str]] = rules or []

        self._selected: str = default_preset
        self._knn: KNeighborsClassifier | None = None

    # ---- YAML ローダ ---------------------------------------------------------
    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "ToneShaper":
        import yaml

        path = Path(path)
        if not path.is_file():
            return cls()

        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}

        presets_raw = data.get("presets") or {}
        levels_raw  = data.get("levels") or {}
        ir_map      = data.get("ir")     or {}
        rules       = data.get("rules")  or []

        preset_map: dict[str, dict[str, int]] = {}
        for name, val in presets_raw.items():
            entry = {"amp": int(val)}
            if name in levels_raw:
                entry.update({k: int(v) for k, v in levels_raw[name].items()})
            preset_map[name] = entry

        return cls(preset_map=preset_map, ir_map=ir_map, rules=rules)

    # ------------------------------------------------------
    # choose_preset
    # ------------------------------------------------------
    def choose_preset(
        self,
        amp_preset: str | None = None,
        intensity: str | None = None,
        avg_velocity: float = 64.0,
    ) -> str:
        """
        プリセット決定アルゴリズム（優先順位）  
        1. amp_preset が明示指定されていればそれを採用  
        2. ルールベース（self.rules）で成立する条件があれば採用  
        3. PRESET_TABLE に (intensity, loud/soft) があれば採用  
        4. codex 版ヒューリスティックで drive / crunch / clean を決定  
        5. 未登録なら default_preset
        """
        # --- 1) 明示指定 ------------------------------------------------------
        if amp_preset:
            chosen = amp_preset
        else:
            lvl = str(intensity or "medium").lower()
            env = {
                "intensity": lvl,
                "avg_vel": avg_velocity,
                "avg_velocity": avg_velocity,
            }

            # --- 2) ルールベース ----------------------------------------------
            chosen: str | None = None
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

            # --- 3) PRESET_TABLE ---------------------------------------------
            if not chosen:
                vel_bucket = "loud" if avg_velocity >= 60 else "soft"
                int_bucket = (
                    "high"   if lvl in {"very_high", "high"} else
                    "medium" if lvl.startswith("medium") else
                    "low"
                )
                chosen = PRESET_TABLE.get((int_bucket, vel_bucket))

            # --- 4) codex ヒューリスティック ---------------------------------
            if not chosen:
                if lvl == "high" or avg_velocity > 100:
                    chosen = "drive"
                elif lvl == "medium" or avg_velocity > 60:
                    chosen = "crunch"
                else:
                    chosen = "clean"

        # --- 5) フォールバック ----------------------------------------------
        if chosen not in self.preset_map:
            chosen = self.default_preset

        self._selected = chosen
        return chosen

    # ------------------------------------------------------
    # CC events
    # ------------------------------------------------------
    def to_cc_events(
        self,
        offset_ql: float = 0.0,
        cc_amp: int = 31,
        cc_rev: int = 91,
        cc_cho: int = 93,
        cc_del: int = 94,
        *,
        as_dict: bool = False,
    ) -> list[tuple[float, int, int]] | list[dict[str, int | float]]:
        """
        選択中プリセットに対応する CC イベントを生成する。

        Returns
        --------
        list[tuple]  : (offset, cc#, value)  
        list[dict]   : {"time": offset, "cc": cc#, "val": value}  (as_dict=True)
        """
        preset = (
            self.preset_map.get(self._selected)
            or self.preset_map.get(self.default_preset, {})
        )

        amp = max(0, min(127, int(preset.get("amp", 0))))
        base = amp
        rev = max(0, min(127, int(preset.get("reverb", int(base * 0.30)))))
        cho = max(0, min(127, int(preset.get("chorus", int(base * 0.30)))))
        dly = max(0, min(127, int(preset.get("delay",  int(base * 0.30)))))

        events: list[tuple[float, int, int]] = [
            (float(offset_ql), cc_amp, amp),
            (float(offset_ql), cc_rev, rev),
            (float(offset_ql), cc_cho, cho),
            (float(offset_ql), cc_del, dly),
        ]
        if as_dict:
            return [{"time": o, "cc": c, "val": v} for o, c, v in events]
        return events

    # ------------------------------------------------------
    # KNN (MFCC → preset)  optional
    # ------------------------------------------------------
    def fit(self, preset_samples: dict[str, NDArray[np.floating]]) -> None:
        """Fit KNN model from preset MFCC samples (optional)."""
        if KNeighborsClassifier is None:  # pragma: no cover
            import warnings
            warnings.warn("scikit-learn not installed; ToneShaper KNN disabled", RuntimeWarning)
            return

        X, y = [], []
        for name, mfcc in preset_samples.items():
            arr = np.asarray(mfcc)
            if arr.ndim != 2:
                raise ValueError("MFCC array must be 2D")
            X.append(arr.mean(axis=1))
            y.append(name)

        self._knn = KNeighborsClassifier(n_neighbors=1)
        self._knn.fit(X, y)

    def predict_preset(self, mfcc: NDArray[np.floating]) -> str:
        """Return preset name predicted from MFCC (optional)."""
        import warnings

        if self._knn is None or KNeighborsClassifier is None:  # pragma: no cover
            warnings.warn("ToneShaper KNN not available; returning default", RuntimeWarning)
            return self.default_preset

        feat = np.asarray(mfcc)
        if feat.ndim != 2:
            raise ValueError("MFCC array must be 2D")
        return str(self._knn.predict([feat.mean(axis=1)])[0])


__all__ = ["ToneShaper"]
