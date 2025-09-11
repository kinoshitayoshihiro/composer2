from __future__ import annotations

"""Split a MIDI file list into N shard files for robust batch processing.

Usage:
  python -m scripts.pedal_split_filelist \
    --file-list tmp/pedal_files.txt \
    --out-dir tmp/pedal_shards \
    --shards 16 --prefix list_
"""

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--file-list", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--shards", type=int, default=8)
    ap.add_argument("--prefix", type=str, default="list_")
    args = ap.parse_args(argv)

    lines = [l.strip() for l in args.file_list.read_text().splitlines() if l.strip()]
    if not lines:
        print("file-list is empty")
        return 0
    s = max(1, int(args.shards))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    # contiguous split (like split -n l/N)
    n = len(lines)
    size = (n + s - 1) // s
    out_paths = []
    for i in range(s):
        chunk = lines[i * size : (i + 1) * size]
        if not chunk:
            continue
        p = args.out_dir / f"{args.prefix}{i:03d}"
        p.write_text("\n".join(chunk) + "\n")
        out_paths.append(p)
    print({"files": len(lines), "shards": len(out_paths), "out_dir": str(args.out_dir)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

