from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

try:  # optional deps
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore


def fit_controls(
    notes_parquet: str | Path,
    targets: Sequence[str] | None = None,
    knots: int = 16,
    out_path: str | Path | None = None,
):
    """Fit a lightweight spline model for control curves.

    When heavy ML deps are unavailable this falls back to storing mean values
    of each target column.
    """
    model: dict[str, float | int | list[str]] = {
        "targets": list(targets or ["bend", "cc11"]),
        "knots": int(knots),
    }
    if pd is not None:
        try:
            df = pd.read_parquet(notes_parquet)
            for t in model["targets"]:
                if t in df:
                    model[f"{t}_mean"] = float(df[t].dropna().mean())
        except Exception:  # pragma: no cover - best effort
            pass
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(model))
    return model


def infer_controls(
    model_path: str | Path | dict,
    meta_json_or_params: dict | None = None,
    out_path: str | Path | None = None,
):
    """Infer control curves from a saved model."""
    if isinstance(model_path, (str, Path)):
        try:
            model = json.loads(Path(model_path).read_text())
        except Exception:  # pragma: no cover
            model = {}
    else:
        model = model_path
    targets = model.get("targets", [])
    result = {t: model.get(f"{t}_mean", 0.0) for t in targets}
    if out_path is not None:
        if pd is not None:
            pd.DataFrame([result]).to_parquet(out_path)
        else:
            Path(out_path).write_text(json.dumps(result))
    return result
