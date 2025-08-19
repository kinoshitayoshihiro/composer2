"""Optional NumPy alias shims."""

import os

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

