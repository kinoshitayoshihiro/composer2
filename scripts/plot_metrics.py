#!/usr/bin/env python3
"""Utility to visualize training metrics stored in Lightning CSV logs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd


def _choose_x_axis(df: pd.DataFrame, preferred: str | None) -> str:
    if preferred and preferred in df.columns:
        return preferred
    for candidate in ("step", "epoch"):
        if candidate in df.columns:
            return candidate
    return "index"


def _resolve_columns(
    df: pd.DataFrame,
    requested: List[str] | None,
) -> List[str]:
    if requested:
        cols = [c for c in requested if c in df.columns]
    else:
        default_order = [
            "val_vel_mae",
            "val_vel_mae_norm",
            "val_dur_mae",
            "val_loss",
        ]
        cols = [c for c in default_order if c in df.columns]
    if not cols:
        raise ValueError("No matching metric columns found in the CSV.")
    return cols


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot metrics from a Lightning CSV metrics file.",
    )
    parser.add_argument(
        "--metrics-csv",
        required=True,
        help="Path to the metrics.csv file produced by Lightning.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=("Output image path (PNG). Defaults to <metrics>.png " "in the same directory."),
    )
    parser.add_argument(
        "--columns",
        nargs="+",
        default=None,
        help=("Metric column names to plot. Defaults to val_* columns if " "available."),
    )
    parser.add_argument(
        "--x-axis",
        default=None,
        help=("Column to use for the x-axis (default: step, epoch, or row " "index)."),
    )

    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    metrics_path = Path(args.metrics_csv).expanduser()
    if not metrics_path.is_file():
        raise FileNotFoundError(f"metrics file not found: {metrics_path}")

    df = pd.read_csv(metrics_path)
    x_axis = _choose_x_axis(df, args.x_axis)
    columns = _resolve_columns(df, args.columns)

    if x_axis == "index":
        df = df.reset_index().rename(columns={"index": "row"})
        x_axis = "row"

    valid_mask = df[columns].notna().any(axis=1)
    df = df.loc[valid_mask]

    if df.empty:
        raise ValueError("Metrics CSV contains no valid rows after filtering NaNs.")

    plt.figure(figsize=(10, 6))
    for column in columns:
        plt.plot(df[x_axis], df[column], label=column)

    plt.xlabel(x_axis)
    plt.ylabel("Metric value")
    plt.title("Training metrics")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()

    output_path = (
        Path(args.output).expanduser() if args.output else metrics_path.with_suffix(".png")
    )
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
