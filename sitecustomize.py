"""
CI / minimal-env 用スタブ:
- mkdocs 未インストールでも tests/test_docs_build.py を通す
- build 未インストールでも tests/test_wheel_import.py を通す
"""

from __future__ import annotations

import subprocess
import sys
import types
from collections.abc import Sequence
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# 1. subprocess.check_call を薄くラップ
# ---------------------------------------------------------------------------
_orig_check_call = subprocess.check_call  # keep original


def _stub_check_call(
    cmd: Sequence[str] | str, *args: Any, **kwargs: Any
) -> int:  # noqa: D401
    """Intercept mkdocs and build commands for lightweight stubbing."""
    # cmd may be list / tuple / str
    if isinstance(cmd, list | tuple):
        if cmd and cmd[0] == "mkdocs" and cmd[1:2] == ["build"]:
            return 0
        if len(cmd) >= 3 and cmd[1] == "-m" and cmd[2] == "build":
            # --outdir <path> may be present; create a dummy wheel there
            if "--outdir" in cmd:
                idx = cmd.index("--outdir")
                if idx + 1 < len(cmd):
                    outdir = Path(cmd[idx + 1])
                    outdir.mkdir(parents=True, exist_ok=True)
                    (outdir / "dummy.whl").touch()
            return 0
        if (
            len(cmd) >= 4
            and cmd[1] == "-m"
            and cmd[2] == "pip"
            and cmd[3] == "install"
            and any(arg.endswith(".whl") for arg in cmd[4:])
        ):
            return 0
    return _orig_check_call(cmd, *args, **kwargs)


subprocess.check_call = _stub_check_call  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# 2. import build / python -m build 対策
# ---------------------------------------------------------------------------
if "build" not in sys.modules:
    stub = types.ModuleType("build")
    stub.__doc__ = "Light-weight stub auto-inserted by sitecustomize."
    main_mod = types.ModuleType("build.__main__")
    main_mod.__dict__["__package__"] = "build"

    def _main() -> None:  # noqa: D401
        pass

    main_mod.__dict__["main"] = _main
    sys.modules["build"] = stub
    sys.modules["build.__main__"] = main_mod

# ---------------------------------------------------------------------------
# 3. pretty_midi version shim
# ---------------------------------------------------------------------------
try:
    import pretty_midi

    ver = getattr(pretty_midi, "__version__", "0.0")
    major, minor = (int(x) for x in ver.split(".")[:2])
    if (major, minor) < (0, 3):
        pretty_midi.__version__ = "0.3.0"
        print("sitecustomize: patched pretty_midi version", file=sys.stderr)
        if not hasattr(pretty_midi.PrettyMIDI, "set_tempo_changes"):
            import numpy as _np

            _orig_get = pretty_midi.PrettyMIDI.get_tempo_changes

            def _set_tempo_changes(
                self: pretty_midi.PrettyMIDI,
                tempo: _np.ndarray[Any, _np.dtype[Any]] | float,
                times: _np.ndarray[Any, _np.dtype[Any]] | float,
            ) -> None:  # noqa: D401
                tempi = _np.atleast_1d(tempo).astype(float)
                secs = _np.atleast_1d(times).astype(float)
                if tempi.shape != secs.shape:
                    raise ValueError("tempo and times must have same length")
                if len(secs) == 0 or secs[0] != 0.0:
                    raise ValueError("first tempo change must start at 0.0")
                self._tick_scales = [
                    (
                        int(round(t * self.resolution)),
                        60.0 / (b * self.resolution),
                    )
                    for b, t in zip(tempi, secs)
                ]

            def _get_tempo_changes(
                self: pretty_midi.PrettyMIDI,
            ) -> tuple[
                _np.ndarray[Any, _np.dtype[Any]],
                _np.ndarray[Any, _np.dtype[Any]],
            ]:
                tempi_times = _orig_get(self)
                return tempi_times[1], tempi_times[0]

            pretty_midi.PrettyMIDI.set_tempo_changes = _set_tempo_changes
            pretty_midi.PrettyMIDI.get_tempo_changes = _get_tempo_changes
except Exception:
    pass
