from __future__ import annotations

import json
import logging
from pathlib import Path

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

# ------------------------------------------------------------
# Default preset library (name -> metadata)
# ------------------------------------------------------------
PRESET_LIBRARY: dict[str, dict] = {
    "clean": {
        "ir_file": "data/ir/clean.wav",
        "gain_db": 0.0,
        "cc_map": {31: 20},
    },
    "crunch": {
        "ir_file": "data/ir/crunch.wav",
        "gain_db": -1.5,
        "cc_map": {31: 50},
    },
    "drive": {
        "ir_file": "data/ir/drive.wav",
        "gain_db": -3.0,
        "cc_map": {31: 90},
    },
    "fuzz": {
        "ir_file": "data/ir/fuzz.wav",
        "gain_db": -6.0,
        "cc_map": {31: 110},
    },
    # ----- Piano/EP Tone Presets ---------------------------------------
    "grand_clean": {
        "ir_file": "irs/german_grand.wav",
        "eq": {200: -1, 3000: 2},
        "cc_map": {31: 25},
    },
    "upright_mellow": {
        "ir_file": "irs/upright_1960.wav",
        "eq": {"lpf": 8000, 120: 1},
        "cc_map": {31: 35},
    },
    "ep_phase": {
        "ir_file": "irs/dx7_ep.wav",
        "chorus_depth": 0.3,
        "cc_map": {31: 55, 94: 30},
    },
}


def _merge_preset_dict(dest: dict, src: dict) -> None:
    """Helper to merge single preset dictionaries."""
    if "ir_file" in src:
        dest["ir_file"] = src["ir_file"]
    if "gain_db" in src:
        dest["gain_db"] = float(src["gain_db"])
    if "cc" in src:
        dest.setdefault("cc_map", {}).update({int(k): int(v) for k, v in src["cc"].items()})
    if "cc_map" in src:
        dest.setdefault("cc_map", {}).update({int(k): int(v) for k, v in src["cc_map"].items()})


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
        self._user_map = preset_map is not None
        self.preset_map: dict[str, dict[str, int]] = {"clean": {"amp": 20}}
        if preset_map:
            for name, data in preset_map.items():
                if isinstance(data, int):
                    self.preset_map[name] = {"amp": int(data)}
                else:
                    self.preset_map[name] = {k: int(v) for k, v in data.items()}

        # merge presets from global library if no user map provided
        if not self._user_map:
            for name, entry in PRESET_LIBRARY.items():
                if name not in self.preset_map and isinstance(entry.get("cc_map"), dict):
                    conv: dict[str, int] = {}
                    for cc, val in entry["cc_map"].items():
                        if int(cc) == 31:
                            conv["amp"] = int(val)
                        elif int(cc) == 91:
                            conv["reverb"] = int(val)
                        elif int(cc) == 93:
                            conv["delay"] = int(val)
                        elif int(cc) == 94:
                            conv["chorus"] = int(val)
                    self.preset_map[name] = conv

        self.ir_map: dict[str, Path] = {k: Path(v) for k, v in (ir_map or {}).items()}
        if not self._user_map:
            for name, entry in PRESET_LIBRARY.items():
                if name not in self.ir_map and entry.get("ir_file"):
                    self.ir_map[name] = Path(entry["ir_file"])
        self.default_preset: str = default_preset
        self.rules: list[dict[str, str]] = rules or []

        # 旧シンプル API と互換のため
        self.presets: dict[str, int] = {
            n: d.get("amp", 0) for n, d in self.preset_map.items()
        }

        self._selected: str = default_preset
        self.last_intensity: str | None = None
        self._knn: KNeighborsClassifier | None = None
        self.fx_envelope: list[dict[str, int | float]] = []

    # ---- YAML ローダ ---------------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str | Path) -> ToneShaper:
        """Load preset and IR mappings from ``path``."""
        import yaml

        path = Path(path)
        if not path.is_file() or path.suffix.lower() not in {".yml", ".yaml"}:
            raise FileNotFoundError(str(path))

        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}

        if "presets" not in data or ("ir" not in data and "rules" not in data):
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
    # preset registry loader
    # ------------------------------------------------------
    @staticmethod
    def load_presets(path: str | None) -> None:
        """Merge presets from ``path`` into :data:`PRESET_LIBRARY`."""
        if not path:
            return

        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(str(p))

        if p.suffix.lower() in {".yml", ".yaml"}:
            import yaml

            data = yaml.safe_load(p.read_text()) or {}
        else:
            data = json.loads(p.read_text())

        if not isinstance(data, dict):
            raise ValueError("Preset file must contain a mapping")

        for name, cfg in data.items():
            if not isinstance(cfg, dict):
                raise ValueError(f"Invalid preset entry for {name}")
            dest = PRESET_LIBRARY.setdefault(name, {})
            _merge_preset_dict(dest, cfg)

    # ------------------------------------------------------
    # choose_preset
    # ------------------------------------------------------
    def choose_preset(
        self,
        amp_hint: str | None = None,
        intensity: str | None = None,
        avg_velocity: float | None = None,
        *,
        style: str | None = None,
        avg_vel: int | None = None,
    ) -> str:
        """
        Select amp preset based on intensity & velocity.

        1. `amp_hint` 明示指定を最優先
        2. rule ベース (`self.rules`)
        3. PRESET_TABLE (threshold 65)
        4. fallback heuristic
        """
        if avg_velocity is None and avg_vel is not None:
            avg_velocity = float(avg_vel)
        avg_velocity = 64.0 if avg_velocity is None else float(avg_velocity)

        if style and not amp_hint:
            amp_hint = style

        # 1) explicit
        chosen: str | None = None
        if amp_hint is not None:
            if amp_hint in self.preset_map or amp_hint in PRESET_LIBRARY:
                chosen = amp_hint
            else:
                chosen = f"{amp_hint}_default"
                if chosen not in self.preset_map:
                    self.preset_map[chosen] = self.preset_map.get(
                        self.default_preset, {"amp": 80}
                    )

        lvl_raw = (intensity or "medium").lower()
        lvl = lvl_raw if lvl_raw in {"low", "medium", "high"} else ""

        # 2) rule-based
        if not chosen and self.rules:
            env = {"intensity": lvl, "avg_velocity": avg_velocity}
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
            if lvl:
                vel_bucket = "loud" if avg_velocity >= 65 else "soft"
                int_bucket = (
                    "high" if lvl.startswith("h") else "medium" if lvl.startswith("m") else "low"
                )
                chosen = PRESET_TABLE.get((int_bucket, vel_bucket))
                if not chosen:
                    chosen = "drive"
                elif chosen not in self.preset_map and self._user_map:
                    chosen = "drive"
            else:
                chosen = "clean"

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
            if chosen in PRESET_LIBRARY and "cc_map" in PRESET_LIBRARY[chosen]:
                conv: dict[str, int] = {}
                for cc, val in PRESET_LIBRARY[chosen]["cc_map"].items():
                    if int(cc) == 31:
                        conv["amp"] = int(val)
                    elif int(cc) == 91:
                        conv["reverb"] = int(val)
                    elif int(cc) == 93:
                        conv["delay"] = int(val)
                    elif int(cc) == 94:
                        conv["chorus"] = int(val)
                self.preset_map[chosen] = conv
            else:
                self.preset_map.setdefault(chosen, {"amp": 80})

        # ensure legacy preset exists
        self.preset_map.setdefault("drive", {"amp": 80})
        self.preset_map.setdefault("drive_default", self.preset_map["drive"])

        self._selected = chosen
        self.last_intensity = lvl or intensity or "medium"
        return chosen


    # ------------------------------------------------------
    # シンプル CC31 だけ返す互換 API
    # ------------------------------------------------------
    def to_cc_events_simple(
        self, preset_name: str, offset_ql: float = 0.0
    ) -> list[dict]:
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
        amp_name: str | None = None,
        intensity: str | None = None,
        mix: float = 1.0,
        *,
        as_dict: bool = False,
        store: bool = True,
    ) -> list[tuple[float, int, int]] | list[dict[str, int | float]]:
        """Return CC events for ``amp_name`` scaled by ``intensity`` and ``mix``.

        When ``amp_name`` or ``intensity`` is ``None`` the values from the last
        :meth:`choose_preset` call are used.
        """

        amp_name = amp_name or self._selected
        intensity = intensity or self.last_intensity or "medium"

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

    def get_ir_file(self, preset_name: str | None = None) -> Path | None:
        """Return IR file for ``preset_name`` or current selection."""
        name = preset_name or self._selected
        ir = self.ir_map.get(name)
        if ir is not None:
            return ir
        entry = PRESET_LIBRARY.get(name)
        if entry and entry.get("ir_file"):
            return Path(entry["ir_file"])
        return None

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
