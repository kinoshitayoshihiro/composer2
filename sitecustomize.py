# Minimal NumPy dtype alias shim for NumPy>=1.24
# This file is auto-imported by Python when present on sys.path.
# It safely restores removed aliases only if they are missing.

from __future__ import annotations

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - numpy may be absent
    np = None  # type: ignore

if np is not None:  # pragma: no cover - trivial glue
    _aliases = {
        "int": int,
        "bool": bool,
        "float": float,
        "complex": complex,
        "object": object,
        "str": str,
    }
    for _name, _typ in _aliases.items():
        # Only patch if attribute is missing (NumPy>=1.24).
        if not hasattr(np, _name):
            try:
                setattr(np, _name, _typ)
            except Exception:
                # Be conservative: never raise at import time
                pass

