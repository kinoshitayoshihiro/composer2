from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd

KEY_COLS = ["track_id", "onset", "pitch"]
OUT_COLS = ["track_id", "onset", "duration", "velocity", "pedal_state", "articulation_label"]


def _load_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in KEY_COLS if c not in df.columns]
    if missing:  # pragma: no cover - validated in CLI
        raise ValueError(f"{path} missing columns: {missing}")
    return df


def merge_artic_features(
    note_features: Iterable[Path],
    pedal_features: Iterable[Path],
    *,
    float_rnd: float = 1e-6,
) -> pd.DataFrame:
    """Merge note and pedal feature CSVs."""
    data: dict[tuple[int, float, int], dict[str, object]] = {}

    def add_row(row: Mapping[str, object]) -> None:
        key = (int(row["track_id"]), float(row["onset"]), int(row["pitch"]))
        rec = data.setdefault(
            key,
            {
                "track_id": int(row["track_id"]),
                "onset": float(row["onset"]),
                "pitch": int(row["pitch"]),
                "duration": 0.0,
                "velocity": 0,
                "pedal_state": 0,
                "articulation_label": "",
            },
        )
        for col, val in row.items():
            if col in KEY_COLS:
                continue
            if col in rec and col != "articulation_label" and rec[col] not in (0, 0.0, "") and val != rec[col]:
                raise ValueError(f"collision for {key} column {col}")
            rec[col] = val

    for path in note_features:
        df = _load_df(path)
        for row in df.to_dict(orient="records"):
            add_row(row)

    for path in pedal_features:
        df = _load_df(path)
        for row in df.to_dict(orient="records"):
            add_row(row)

    merged = pd.DataFrame(list(data.values()))
    merged = merged.sort_values(KEY_COLS).reset_index(drop=True)
    dec = int(round(-math.log10(float_rnd))) if float_rnd > 0 else 0
    merged["onset"] = merged["onset"].round(dec)
    merged["duration"] = merged["duration"].round(dec)
    return merged


def split_artic_features(
    df: pd.DataFrame,
    note_map: Mapping[Path, Iterable[str]],
    pedal_map: Mapping[Path, Iterable[str]],
) -> None:
    """Write feature CSVs from merged dataframe."""
    base = KEY_COLS
    for path, cols in note_map.items():
        df[base + list(cols)].to_csv(path, index=False)
    for path, cols in pedal_map.items():
        df[base + list(cols)].to_csv(path, index=False)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Merge articulation feature CSVs")  # pragma: no cover - CLI setup
    ap.add_argument("--note-features", nargs="+", type=Path, required=True)  # pragma: no cover - CLI setup
    ap.add_argument("--pedal-features", nargs="*", type=Path, default=[])  # pragma: no cover - CLI setup
    ap.add_argument("--out", type=Path, default=Path("artic.csv"))  # pragma: no cover - CLI setup
    ap.add_argument("--float-rnd", type=float, default=1e-6)  # pragma: no cover - CLI setup
    ap.add_argument("--schema-check", action="store_true")  # pragma: no cover - CLI setup
    ns = ap.parse_args(argv)  # pragma: no cover - CLI setup

    df = merge_artic_features(  # pragma: no cover - CLI execution
        ns.note_features, ns.pedal_features, float_rnd=ns.float_rnd
    )
    df_out = df.drop(columns=["pitch"])  # pragma: no cover - CLI execution
    if ns.schema_check and list(df_out.columns) != OUT_COLS:  # pragma: no cover - CLI execution
        ap.error(f"output columns {list(df_out.columns)} != {OUT_COLS}")
    df_out.to_csv(ns.out, index=False)  # pragma: no cover - CLI execution
    return 0  # pragma: no cover - CLI execution


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
