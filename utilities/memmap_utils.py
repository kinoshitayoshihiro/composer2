from __future__ import annotations

from pathlib import Path

import numpy as np


def load_memmap(path: Path, *, shape: tuple[int, ...]) -> np.memmap:
    """Return a read-only memmap view of ``path``.

    Parameters
    ----------
    path:
        Location of the ``.mmap`` file.
    shape:
        Shape of the stored array.
    """
    return np.memmap(path, dtype="float32", mode="r", shape=shape)

