# ruff: noqa: E402, F404
# === local-stub (CLI) — precedes everything ===  # pragma: no cover
import sys
import types

for _n in ("yaml", "pkg_resources"):
    sys.modules.setdefault(_n, types.ModuleType(_n))
# === end stub ===

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch.multiprocessing as mp

from utilities.bass_timbre_dataset import compute_mel_pair, find_pairs


def _process(args: tuple) -> tuple[str, str, str, int]:
    pair, src_suffix, max_len, out_dir = args
    mel_src, mel_tgt = compute_mel_pair(
        pair.src_path, pair.tgt_path, pair.midi_path, max_len
    )
    out = out_dir / f"{pair.id}__{src_suffix}->{pair.tgt_suffix}.npy"
    np.save(out, np.stack([mel_src, mel_tgt]))
    return pair.id, src_suffix, pair.tgt_suffix, mel_src.shape[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract timbre dataset")
    parser.add_argument("--in_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--src", type=str, default="wood")
    parser.add_argument("--tgt", type=str, nargs="+", required=True)
    parser.add_argument("--max_len", type=int, default=30_000)
    parser.add_argument("--num_workers", type=int, default=1)
    args = parser.parse_args()

    pairs = find_pairs(args.in_dir, args.src, args.tgt)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with mp.Pool(args.num_workers) as pool:
        results = pool.map(
            _process,
            [(p, args.src, args.max_len, args.out_dir) for p in pairs],
        )

    manifest = args.out_dir / "dataset.csv"
    with manifest.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id", "src", "tgt", "n_frames"])
        for row in results:
            writer.writerow(row)
    print(f"Wrote {manifest}")


if __name__ == "__main__":
    main()
