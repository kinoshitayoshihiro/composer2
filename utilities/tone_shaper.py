from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
from numpy.typing import NDArray

try:
    from sklearn.neighbors import KNeighborsClassifier  # type: ignore
except Exception:  # pragma: no cover – optional dependency
    KNeighborsClassifier = None  # type: ignore

# ──────────────────────────────────────────────────────────
# プリセット表 (Intensity × Loud/Soft)
# ──────────────────────────────────────────────────────────
PRESET_TABLE: dict[tuple[str, str], str] = {
    ("low", "soft"): "clean",
    ("low", "loud"): "crunch",
    ("medium", "soft"): "crunch",
    ("medium", "loud"): "drive",
    ("high", "soft"): "drive",
    ("high", "loud"): "fuzz",
}

logger = logging.getLogger(__name__)


class ToneShaper:
    """
    Amp / Cabinet プリセットを選択し、必要な CC イベントを生成するユーティリティ。

    - **プリセットマップ** : プリセット名 → {"amp": 0-127, "reverb": …, …}
    - **IR マップ**       : プリセット名 → Impulse Response ファイルパス
    - **ルール**          : ``{"if": "<python-expr>", "preset": "<name>"}``
    - **KNN**             : MFCC からプリセット推定 (任意)
    """

    # ------------------------------------------------------
    # constructor / loader
    # ------------------------------------------------------
    def __init__(
        self,
        preset_map: dict[str, dict[str, int] | int] | None = None,
        ir_map: dict[str, Path] | None = None,
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

        self.ir_map: dict[str, Path] = {k: Path(v) for k, v in (ir_map or {}).items()}
        self.default_preset: str = default_preset
        self.rules: list[dict[str, str]] = rules or []

        # 旧シンプル API と互換のため
        self.presets: Dict[str, int] = {
            n: d.get("amp", 0) for n, d in self.preset_map.items()
        }

        self._selected: str = default_preset
        self._knn: KNeighborsClassifier | None = None
        self.fx_envelope: list[dict[str, int | float]] = []

    # ---- YAML ローダ ---------------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str | Path) -> "ToneShaper":
        """Load preset and IR mappings from ``path``."""
        import yaml

        path = Path(path)
        if not path.is_file() or path.suffix.lower() not in {".yml", ".yaml"}:
            raise FileNotFoundError(str(path))

        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}

        if not {"presets", "ir"}.issubset(data.keys()):
            raise ValueError("Malformed preset file")

        presets_raw = data.get("presets", {})
        levels_raw = data.get("levels", {})
        ir_raw = data.get("ir", {})
        rules = data.get("rules", [])

        preset_map: dict[str, dict[str, int]] = {}
        for name, val in presets_raw.items():
            try:
                amp_val = int(val)
            except Exception as exc:
                raise ValueError(f"Invalid preset value for {name}: {val}") from exc
            if not 0 <= amp_val <= 127:
                raise ValueError(f"Preset value for {name} out of range: {amp_val}")
            entry = {"amp": amp_val}
            if name in levels_raw:
                tmp: dict[str, int] = {}
                for k, v in levels_raw[name].items():
                    try:
                        iv = int(v)
                    except Exception as exc:
                        raise ValueError(f"Invalid level value for {name}:{k}") from exc
                    if not 0 <= iv <= 127:
                        raise ValueError(f"Level value for {name}:{k} out of range: {iv}")
                    tmp[k] = iv
                entry.update(tmp)
            preset_map[name] = entry

        ir_map: dict[str, Path] = {}
        for name, path_str in ir_raw.items():
            p = Path(path_str)
            if not p.is_file():
                logger.warning("IR file missing: %s", path_str)
            ir_map[name] = p

        return cls(preset_map=preset_map, ir_map=ir_map, rules=rules)

    # ------------------------------------------------------
    # choose_preset
    # ------------------------------------------------------
    def choose_preset(
        self,
        amp_hint: str | None = None,
        intensity: str | None = None,
        avg_velocity: float | None = None,
    ) -> str:
        """
        Select amp preset based on intensity & velocity.

        1. `amp_hint` 明示指定を最優先
        2. rule ベース (`self.rules`)
        3. PRESET_TABLE (threshold 65)
        4. fallback heuristic
        """
        avg_velocity = 64.0 if avg_velocity is None else float(avg_velocity)

        # 1) explicit
        chosen: str | None = amp_hint

        lvl = (intensity or "medium").lower()

        # 2) rule-based
        if not chosen and self.rules:
            env = {"intensity": lvl, "avg_vel": avg_velocity}
            for rule in self.rules:
                cond, preset = rule.get("if"), rule.get("preset")
                if not cond or not preset:
                    continue
                try:
                    if eval(cond, {"__builtins__": {}}, env):  # nosec
                        chosen = preset
                        break
                except Exception:
                    continue

        # 3) table
        if not chosen:
            vel_bucket = "loud" if avg_velocity >= 65 else "soft"
            int_bucket = (
                "high"
                if lvl.startswith("h")
                else "medium" if lvl.startswith("m") else "low"
            )
            chosen = PRESET_TABLE.get((int_bucket, vel_bucket))
            if chosen and chosen not in self.preset_map:
                chosen = None

        # 4) fallback
        if not chosen:
            chosen = (
                "drive"
                if avg_velocity > 100 or lvl == "high"
                else "crunch"
                if avg_velocity > 60 or lvl == "medium"
                else "clean"
            )

        if not chosen:
            chosen = self.default_preset

        if chosen not in self.preset_map:
            logger.warning("Unknown preset %s; fallback to default", chosen)
            chosen = self.default_preset

        self._selected = chosen
        return chosen

    # ------------------------------------------------------
    # deprecated wrapper for old signature
    # ------------------------------------------------------
    @staticmethod
    def _deprecated(func):
        import functools, warnings

        @functools.wraps(func)
        def wrapper(*a, **k):
            warnings.warn("choose_preset legacy signature is deprecated", DeprecationWarning, stacklevel=2)
            return func(*a, **k)

        return wrapper

    @_deprecated
    def choose_preset_legacy(
        self,
        amp_preset: str | None = None,
        intensity: str | None = None,
        avg_velocity: float = 64.0,
    ) -> str:
        return self.choose_preset(amp_preset, intensity, avg_velocity)

    # ------------------------------------------------------
    # シンプル CC31 だけ返す互換 API
    # ------------------------------------------------------
    def to_cc_events_simple(
        self, preset_name: str, offset_ql: float = 0.0
    ) -> List[dict]:
        """Return single CC31 event dictionary (旧互換)."""
        value = self.presets.get(preset_name, self.presets[self.default_preset])
        return [{"time": float(offset_ql), "cc": 31, "val": value}]

    # ------------------------------------------------------
    # multi-CC events (amp/rev/cho/dly)
    # ------------------------------------------------------
    def _events_for_selected(
        self,
        offset_ql: float = 0.0,
        cc_amp: int = 31,
        cc_rev: int = 91,
        cc_del: int = 93,
        cc_cho: int = 94,
        *,
        as_dict: bool = False,
        store: bool = True,
    ) -> list[tuple[float, int, int]] | list[dict[str, int | float]]:
        preset = self.preset_map.get(self._selected, self.preset_map[self.default_preset])

        amp = max(0, min(127, int(preset.get("amp", 0))))
        base = amp
        rev = max(0, min(127, int(preset.get("reverb", int(base * 0.30)))))
        cho = max(0, min(127, int(preset.get("chorus", int(base * 0.30)))))
        dly = max(0, min(127, int(preset.get("delay", int(base * 0.30)))))

        events = {
            (float(offset_ql), cc_amp, amp),
            (float(offset_ql), cc_rev, rev),
            (float(offset_ql), cc_del, dly),
            (float(offset_ql), cc_cho, cho),
        }
        if store:
            self.fx_envelope = [
                {"time": t, "cc": c, "val": v} for t, c, v in sorted(events, key=lambda e: e[0])
            ]
        if as_dict:
            return [
                {"time": t, "cc": c, "val": v} for t, c, v in sorted(events, key=lambda e: e[0])
            ]
        return sorted(events, key=lambda e: e[0])

    def to_cc_events(
        self,
        amp_name: str,
        intensity: str,
        mix: float = 1.0,
        *,
        as_dict: bool = False,
        store: bool = True,
    ) -> list[tuple[float, int, int]] | list[dict[str, int | float]]:
        """Return CC events for ``amp_name`` scaled by ``intensity`` and ``mix``."""

        preset = self.preset_map.get(amp_name) or self.preset_map.get(
            self.default_preset, {}
        )
        amp_base = int(preset.get("amp", 0))
        amp = max(0, min(127, amp_base))
        scale = {"low": 0.5, "medium": 0.8, "high": 1.0}.get(intensity.lower(), 0.8)
        rev_base = int(preset.get("reverb", int(amp_base * 0.3)))
        cho_base = int(preset.get("chorus", int(amp_base * 0.3)))
        dly_base = int(preset.get("delay", int(amp_base * 0.3)))
        rev = max(0, min(127, int(rev_base * scale * mix)))
        cho = max(0, min(127, int(cho_base * scale * mix)))
        dly = max(0, min(127, int(dly_base * scale * mix)))

        events = {
            (0.0, 31, amp),
            (0.0, 91, rev),
            (0.0, 93, dly),
            (0.0, 94, cho),
        }
        if store:
            self.fx_envelope = [
                {"time": t, "cc": c, "val": v} for t, c, v in sorted(events, key=lambda e: e[0])
            ]
        if as_dict:
            return [
                {"time": t, "cc": c, "val": v} for t, c, v in sorted(events, key=lambda e: e[0])
            ]
        return sorted(events, key=lambda e: e[0])

    def render_with_ir(self, mix_wav: Path, preset_name: str, out: Path) -> Path:
        """Apply impulse response for ``preset_name`` to ``mix_wav``."""
        ir_path = self.ir_map.get(preset_name)
        if ir_path is None:
            raise KeyError(preset_name)
        from .convolver import render_with_ir as _render

        return _render(mix_wav, ir_path, out)

    # ------------------------------------------------------
    # KNN (MFCC → preset)  optional
    # ------------------------------------------------------
    def fit(self, preset_samples: dict[str, NDArray[np.floating]]) -> None:
        """Fit KNN model from preset MFCC samples (optional)."""
        if KNeighborsClassifier is None:  # pragma: no cover
            import warnings

            warnings.warn(
                "scikit-learn not installed; ToneShaper KNN disabled",
                RuntimeWarning,
            )
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
            warnings.warn(
                "ToneShaper KNN not available; returning default",
                RuntimeWarning,
            )
            return self.default_preset

        feat = np.asarray(mfcc)
        if feat.ndim != 2:
            raise ValueError("MFCC array must be 2D")
        return str(self._knn.predict([feat.mean(axis=1)])[0])


__all__ = ["ToneShaper"]
