from __future__ import annotations

"""Rewrite song_id column in a phrase CSV using a mapping CSV.

- Input CSV must have a header and include a column named `song_id` (or specify --col).
- Mapping CSV must have two columns: key -> value (configurable via --map-key/--map-to).
  Typical usage maps original validation CSV song_id to MIDI filename stems, so that
  per-song thresholds computed on validation can be applied to MIDI inference.

Example:
  python -m tools.rewrite_song_id \
    --in-csv data/phrase_csv/gtr_midi_gap10_valid.csv \
    --map data/phrase_csv/songid_to_midistem.csv \
    --map-key song_id --map-to midi_stem \
    --out-csv data/phrase_csv/gtr_midi_gap10_valid_midistem.csv
"""

import argparse
import csv
from pathlib import Path


def load_map(path: Path, key: str, to: str) -> dict[str, str]:
    with path.open() as f:
        r = csv.DictReader(f)
        if r.fieldnames is None or key not in r.fieldnames or to not in r.fieldnames:
            raise SystemExit(f"mapping CSV must have columns {key!r} and {to!r}")
        m: dict[str, str] = {}
        for row in r:
            k = str(row.get(key, "")).strip()
            v = str(row.get(to, "")).strip()
            if not k:
                continue
            if v:
                m[k] = v
        return m


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-csv", type=Path, required=True)
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("--map", type=Path, required=True, help="CSV with mapping rows")
    ap.add_argument("--map-key", type=str, default="song_id", help="column in map CSV to match input song_id")
    ap.add_argument("--map-to", type=str, default="midi_stem", help="column in map CSV to replace song_id with")
    ap.add_argument("--col", type=str, default="song_id", help="target column to rewrite in input CSV")
    args = ap.parse_args(argv)

    mapping = load_map(args.map, args.map_key, args.map_to)
    rows_out = []
    with args.in_csv.open() as f:
        r = csv.DictReader(f)
        if r.fieldnames is None:
            raise SystemExit("input CSV has no header")
        if args.col not in r.fieldnames:
            raise SystemExit(f"input CSV missing column {args.col!r}")
        fields = r.fieldnames
        missing = 0
        for row in r:
            sid = str(row.get(args.col, ""))
            sid_new = mapping.get(sid)
            if sid_new:
                row[args.col] = sid_new
            else:
                missing += 1
            rows_out.append(row)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows_out)
    print(f"wrote {args.out_csv} rows={len(rows_out)} missing_map={missing}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

