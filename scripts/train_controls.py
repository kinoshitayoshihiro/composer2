"""Minimal spline-based CC trainer."""

import argparse

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional
    pd = None  # type: ignore

from utilities.controls_spline import fit_spline


def main(argv=None):
    parser = argparse.ArgumentParser(description="Fit control splines")
    parser.add_argument("input", help="CSV or Parquet file with 'time' and 'cc11'")
    parser.add_argument("output", help="Output .npz path")
    parser.add_argument("--max-knots", type=int, default=16)
    args = parser.parse_args(argv)

    if pd is None:
        raise SystemExit("pandas is required for this script")
    if args.input.endswith(".parquet"):
        df = pd.read_parquet(args.input)
    else:
        df = pd.read_csv(args.input)
    times = df["time"].to_numpy()
    values = df["cc11"].to_numpy()
    kt, kv = fit_spline(times, values, max_knots=args.max_knots)
    np.savez(args.output, times=kt, values=kv)


if __name__ == "__main__":
    main()
