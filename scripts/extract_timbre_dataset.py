# ruff: noqa: E402,F404
# === local-stub (CLI) — precedes everything ===  # pragma: no cover
import importlib.machinery
import sys
import types

for _n in ("yaml", "pkg_resources", "scipy", "scipy.signal"):
    mod = types.ModuleType(_n)
    mod.__spec__ = importlib.machinery.ModuleSpec(_n, loader=None)
    sys.modules.setdefault(_n, mod)
# === end stub ===
"""CLI utility to generate a manifest of timbre-transfer training pairs."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - hints only
    pass

from utilities.bass_timbre_dataset import BassTimbreDataset, TimbrePair, compute_mel_pair


def _process(args: tuple[TimbrePair, str, int, Path]) -> tuple[str, str, str, int]:
    """Compute and save mel pair to ``out_dir``."""
    pair, src_suffix, max_len, out_dir = args
    mel_src, mel_tgt = compute_mel_pair(pair.src_path, pair.tgt_path, pair.midi_path, max_len)
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
    parser.add_argument("--no_cache", action="store_true", help="disable dataset cache")
    args = parser.parse_args()

    ds = BassTimbreDataset(
        root=args.in_dir,
        src_suffix=args.src,
        tgt_suffixes=args.tgt,
        max_len=args.max_len,
        cache=not args.no_cache,
    )

    if not ds.pairs:
        print("No valid pairs found", file=sys.stderr)
        sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.num_workers == 1:
        try:
            from tqdm import tqdm  # type: ignore[import,unused-ignore]
        except Exception:
            def tqdm(x: list[object] | object) -> list[object] | object:  # pragma: no cover
                return x
        results = []
        for pair in tqdm(ds.pairs):
            results.append(_process((pair, args.src, args.max_len, args.out_dir)))
    else:
        try:
            import torch.multiprocessing as mp
        except Exception as exc:  # pragma: no cover - torch absent
            raise RuntimeError("multiprocessing requires torch") from exc
        with mp.Pool(args.num_workers) as pool:
            results = pool.map(
                _process,
                [(p, args.src, args.max_len, args.out_dir) for p in ds.pairs],
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
