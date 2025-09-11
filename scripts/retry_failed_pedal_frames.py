from __future__ import annotations

"""Retry pedal frame extraction for files listed in a fail-log.

Reads a text file with one MIDI path per line, extracts features via
utilities.pedal_frames.extract_from_midi, and appends to a target CSV.

Examples:
  python -m scripts.retry_failed_pedal_frames \
    --fail-list tmp/pedal_failed.txt \
    --out-csv data/pedal/all.csv \
    --sr 16000 --hop 4096 --cc-th 1 --max-seconds 60
"""

import argparse
from pathlib import Path

import pandas as pd

from utilities.pedal_frames import extract_from_midi, SR as DEFAULT_SR, HOP_LENGTH as DEFAULT_HOP


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fail-list", type=Path, required=True)
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("--sr", type=int, default=DEFAULT_SR)
    ap.add_argument("--hop", type=int, default=DEFAULT_HOP)
    ap.add_argument("--cc-th", type=int, default=64)
    ap.add_argument("--max-seconds", type=float, default=0.0)
    args = ap.parse_args(argv)

    paths = [Path(p.strip()) for p in args.fail_list.read_text().splitlines() if p.strip()]
    if not paths:
        print("no paths in fail-list")
        return 0
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    header_written = args.out_csv.exists() and args.out_csv.stat().st_size > 0
    ok = 0
    fail = 0
    for path in paths:
        try:
            df = extract_from_midi(path, sr=args.sr, hop_length=args.hop, cc_th=args.cc_th, max_seconds=args.max_seconds)
            if df.empty:
                ok += 1
                continue
            df["file"] = path.name
            mode = "a" if header_written else "w"
            df.to_csv(args.out_csv, index=False, mode=mode, header=not header_written)
            header_written = True
            ok += 1
        except Exception:
            fail += 1
    print({"retried": len(paths), "ok": ok, "fail": fail, "out": str(args.out_csv)})
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

