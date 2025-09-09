from __future__ import annotations

"""Utilities to sample or shard very large phrase CSVs by song_id without
loading the entire file into memory. Works in a single streaming pass.

Usage examples:

  # Keep ~5% of songs for a small training subset
  python -m tools.shard_phrase_csv sample \
      --in data/phrase_csv/pno_midi_gap10_train.csv \
      --out data/phrase_csv/pno_midi_gap10_train.sample05.csv \
      --ratio 0.05

  # Split into 20 shards by song_id hash (header is written to each shard)
  python -m tools.shard_phrase_csv shard \
      --in data/phrase_csv/pno_midi_gap10_train.csv \
      --out-dir data/phrase_csv/pno_shards/train \
      --shards 20

The sharding is deterministic by song_id: all rows of a song go to the same
output file. This preserves grouping-by-bar semantics within a song.
"""

import argparse
import csv
import hashlib
from pathlib import Path


def _song_key(row: dict[str, str]) -> str:
    sid = row.get("song_id") or row.get("file") or row.get("path") or ""
    return str(sid)


def _hash01(s: str) -> float:
    h = hashlib.md5(s.encode("utf-8", errors="ignore")).hexdigest()
    # use first 16 hex digits -> 64-bit range for speed
    v = int(h[:16], 16)
    return v / float(1 << 64)


def cmd_sample(path_in: Path, path_out: Path, ratio: float) -> int:
    ratio = max(0.0, min(1.0, float(ratio)))
    path_out.parent.mkdir(parents=True, exist_ok=True)
    with path_in.open() as f_in, path_out.open("w", newline="") as f_out:
        r = csv.DictReader(f_in)
        w = csv.DictWriter(f_out, fieldnames=r.fieldnames)
        w.writeheader()
        for row in r:
            sid = _song_key(row)
            if not sid:
                # fallback: keep a tiny fraction to avoid total drop
                if _hash01(str(row)) < ratio * 0.1:
                    w.writerow(row)
                continue
            if _hash01(sid) < ratio:
                w.writerow(row)
    return 0


def cmd_shard(path_in: Path, out_dir: Path, shards: int) -> int:
    n = max(1, int(shards))
    out_dir.mkdir(parents=True, exist_ok=True)
    writers: list[csv.DictWriter] = []
    files = []
    try:
        with path_in.open() as f_in:
            r = csv.DictReader(f_in)
            # prepare shard writers with header
            for i in range(n):
                p = out_dir / f"shard_{i:03d}.csv"
                fp = p.open("w", newline="")
                files.append(fp)
                w = csv.DictWriter(fp, fieldnames=r.fieldnames)
                w.writeheader()
                writers.append(w)
            # stream and dispatch by song hash
            for row in r:
                sid = _song_key(row)
                idx = int(_hash01(sid) * n) if sid else 0
                idx = max(0, min(n - 1, idx))
                writers[idx].writerow(row)
    finally:
        for fp in files:
            try:
                fp.close()
            except Exception:
                pass
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("sample", help="sample a fraction of songs by song_id")
    sp.add_argument("--in", dest="src", type=Path, required=True)
    sp.add_argument("--out", dest="dst", type=Path, required=True)
    sp.add_argument("--ratio", type=float, required=True)
    sh = sub.add_parser("shard", help="split into N shards by song_id")
    sh.add_argument("--in", dest="src", type=Path, required=True)
    sh.add_argument("--out-dir", type=Path, required=True)
    sh.add_argument("--shards", type=int, required=True)
    return p


def main(argv: list[str] | None = None) -> int:
    p = build_parser()
    args = p.parse_args(argv)
    if args.cmd == "sample":
        return cmd_sample(args.src, args.dst, args.ratio)
    if args.cmd == "shard":
        return cmd_shard(args.src, args.out_dir, args.shards)
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

