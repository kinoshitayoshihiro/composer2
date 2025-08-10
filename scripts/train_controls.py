from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train placeholder control spline model")
    parser.add_argument("notes", help="Path to notes.parquet")
    parser.add_argument("-o", "--out", required=True, help="Output model path")
    args = parser.parse_args(argv)

    try:
        import pandas as pd  # type: ignore
    except Exception:  # pragma: no cover - pandas optional
        raise SystemExit("pandas required for training controls")

    _ = pd.read_parquet(args.notes)
    from models.controls_spline import ControlSplineModel

    model = ControlSplineModel()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.out)
    print(f"Saved placeholder model to {args.out}")


if __name__ == "__main__":
    main()
