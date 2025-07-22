"""Project-local stub loaded automatically by Python at start-up.
Silences heavy optional dependencies in minimal CI environments."""

import importlib.util
import os
from pathlib import Path

if os.getenv("COMPOSER_CI_STUBS") == "1":
    path = Path(__file__).resolve().parent / "utilities" / "stub_utils.py"
    spec = importlib.util.spec_from_file_location("_stub_utils", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    module.install_stubs(
        ["pkg_resources", "yaml", "scipy", "scipy.signal", "music21"],
        force=True,
    )
