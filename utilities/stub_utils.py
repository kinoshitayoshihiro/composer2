from __future__ import annotations

import importlib
import subprocess
import sys
import types

STUB_MODULES: dict[str, dict[str, str]] = {}


def install_stub(name: str) -> None:
    """Install *name* via pip or provide a dummy module."""
    try:
        importlib.import_module(name)
        return
    except Exception:
        pass
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", name])
        importlib.import_module(name)
        return
    except Exception:
        mod = types.ModuleType(name)
        sys.modules[name] = mod
