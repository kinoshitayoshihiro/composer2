from __future__ import annotations

import argparse
from pathlib import Path

from ml.controls_spline import fit_controls


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Train lightweight control curves")
    p.add_argument("notes", help="Path to notes parquet file")
    p.add_argument("-o", "--out", required=True, help="Output model path")
    p.add_argument(
        "--targets",
        default="bend,cc11",
        help="Comma-separated list of targets to model",
    )
    p.add_argument("--spline-knots", type=int, default=16, help="Number of spline knots")
    args = p.parse_args(argv)

    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    fit_controls(args.notes, targets=targets, knots=args.spline_knots, out_path=args.out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
