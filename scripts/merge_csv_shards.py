from __future__ import annotations

"""Merge many CSV shard files into one, writing header once.

Usage:
  python -m scripts.merge_csv_shards --glob 'data/pedal/frames_list_*\.csv' --out data/pedal/all_sharded.csv
or
  python -m scripts.merge_csv_shards --inputs data/pedal/frames_a.csv data/pedal/frames_b.csv --out data/pedal/all.csv
"""

import argparse
import glob
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--glob", type=str, help="glob pattern for input CSVs")
    ap.add_argument("--inputs", type=Path, nargs="*", help="explicit input CSV paths")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    inputs: list[str] = []
    if args.glob:
        inputs.extend(sorted(glob.glob(args.glob)))
    if args.inputs:
        inputs.extend([str(p) for p in args.inputs])
    inputs = [p for p in inputs if Path(p).is_file()]
    if not inputs:
        print("no input CSVs")
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    header_written = False
    with args.out.open("w") as out:
        for i, p in enumerate(inputs):
            with open(p, "r") as f:
                for j, line in enumerate(f):
                    if j == 0 and header_written:
                        continue
                    out.write(line)
            header_written = True
    print({"merged": len(inputs), "out": str(args.out)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

