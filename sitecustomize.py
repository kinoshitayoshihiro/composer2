"""Optional runtime tweaks for tests and compatibility."""

import os
import sys

if sys.platform != "win32":
    try:
        import multiprocessing as _mp

        if "fork" in _mp.get_all_start_methods():
            _mp.set_start_method("fork", force=True)
    except Exception:  # pragma: no cover - best effort on unsupported platforms
        pass

if os.getenv("COMPOSER2_ENABLE_NUMPY_SHIM") == "1":
    try:
        import numpy as np  # noqa: F401
        # Only add when missing, avoid shadowing real dtypes.
        if not hasattr(np, "int"):
            np.int = int  # type: ignore[attr-defined]
        if not hasattr(np, "bool"):
            np.bool = bool  # type: ignore[attr-defined]
        if not hasattr(np, "float"):
            np.float = float  # type: ignore[attr-defined]
    except Exception:
        # Be silent: NumPy may not be installed in minimal environments.
        pass

