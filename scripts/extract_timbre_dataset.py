from __future__ import annotations

# ruff: noqa: E402,F404
# === local-stub (CLI) — precedes everything ===  # pragma: no cover
import importlib.machinery
import importlib.util
import sys
import types
from pathlib import Path

# ``utilities`` pulls in heavy optional dependencies on import.  When running the
# CLI in isolated test environments these may be unavailable.  Provide minimal
# stub modules so imports succeed without the full packages being installed.
UTILS_DIR = Path(__file__).resolve().parent.parent / "utilities"
utils_pkg = types.ModuleType("utilities")
utils_pkg.__path__ = [str(UTILS_DIR)]
sys.modules.setdefault("utilities", utils_pkg)

try:  # pragma: no cover - optional dependency
    import scipy.signal  # type: ignore
except Exception:
    for _n in ("pkg_resources", "scipy", "scipy.signal"):
        if _n not in sys.modules:
            mod = types.ModuleType(_n)
            mod.__spec__ = importlib.machinery.ModuleSpec(_n, loader=None)
            sys.modules[_n] = mod
# === end stub ===
"""CLI utility to generate a manifest of timbre-transfer training pairs."""


import argparse
import csv

import numpy as np


# Import ``bass_timbre_dataset`` directly from file to avoid importing the
# heavy ``utilities`` package and its optional dependencies.
module_name = "utilities.bass_timbre_dataset"
_spec = importlib.util.spec_from_file_location(
    module_name,
    UTILS_DIR / "bass_timbre_dataset.py",
)
if _spec is None or _spec.loader is None:  # pragma: no cover - should not happen
    raise RuntimeError("Failed to load bass_timbre_dataset module")
_mod = importlib.util.module_from_spec(_spec)
sys.modules[module_name] = _mod
try:
    _spec.loader.exec_module(_mod)
except ImportError as exc:  # pragma: no cover - optional deps missing
    raise RuntimeError(f"Failed to load {module_name}") from exc
BassTimbreDataset = _mod.BassTimbreDataset
TimbrePair = _mod.TimbrePair
compute_mel_pair = _mod.compute_mel_pair


def _process(args: tuple[TimbrePair, str, int, Path]) -> tuple[str, str, str, int]:
    """Compute and save mel pair to ``out_dir``."""
    pair, src_suffix, max_len, out_dir = args
    mel_src, mel_tgt = compute_mel_pair(
        pair.src_path, pair.tgt_path, pair.midi_path, max_len
    )
    out = out_dir / f"{pair.id}__{src_suffix}->{pair.tgt_suffix}.npy"
    np.save(out, np.stack([mel_src, mel_tgt]), allow_pickle=False)
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

            def tqdm(
                x: list[object] | object,
            ) -> list[object] | object:  # pragma: no cover
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
