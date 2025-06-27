from __future__ import annotations

import pickle
from pathlib import Path
from typing import Sequence

CACHE_NAME = ".loops_cache.pkl"


def scan_loops(loop_dir: Path, exts: Sequence[str], *, use_cache: bool = True) -> list[Path]:
    """Return loop files under ``loop_dir`` matching ``exts``.

    Results are cached to ``CACHE_NAME`` inside ``loop_dir``. The cache is reused
    if it is newer than every file in ``loop_dir``.
    """

    if not loop_dir.exists():
        raise FileNotFoundError(f"loop directory {loop_dir} not found")
    if not exts:
        raise ValueError("no extensions specified")

    normalized = {e.lower().lstrip(".") for e in exts}
    cache_path = loop_dir / CACHE_NAME
    if use_cache and cache_path.exists():
        cache_mtime = cache_path.stat().st_mtime
        if all(p.stat().st_mtime <= cache_mtime for p in loop_dir.iterdir()):
            with cache_path.open("rb") as fh:
                return pickle.load(fh)

    files = [
        p
        for p in loop_dir.iterdir()
        if p.suffix.lower().lstrip(".") in normalized
    ]
    if use_cache:
        with cache_path.open("wb") as fh:
            pickle.dump(files, fh)
    return files
