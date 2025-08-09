from __future__ import annotations

import argparse
import json
from pathlib import Path

from ml.controls_spline import infer_controls


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Infer control curves from a model")
    p.add_argument("-m", "--model", required=True, help="Model path")
    p.add_argument("-o", "--out", required=True, help="Output parquet/JSON path")
    p.add_argument("--meta", help="Optional JSON file with meta parameters")
    args = p.parse_args(argv)

    meta = None
    if args.meta:
        try:
            meta = json.loads(Path(args.meta).read_text())
        except Exception:  # pragma: no cover - best effort
            meta = None
    infer_controls(args.model, meta_json_or_params=meta, out_path=args.out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
