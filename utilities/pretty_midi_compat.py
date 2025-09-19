"""Compat shim for pretty_midi.

Some environments (old forks / stubs) make PrettyMIDI.get_tempo_changes()
return plain Python lists, which breaks pretty_midi.PrettyMIDI.get_end_time()
because it blindly calls `.tolist()`.

By importing this module once at startup, we monkey-patch the method so it
*always* returns numpy.ndarray.
"""

from __future__ import annotations

import os as _os

import numpy as _np
import pretty_midi as _pm

# If someone else already patched it, don't stack.
if not getattr(_pm.PrettyMIDI.get_tempo_changes, "_composer2_patched", False):
    _orig_get_tempo_changes = _pm.PrettyMIDI.get_tempo_changes
    _orig_write = _pm.PrettyMIDI.write

    def _safe_get_tempo_changes(self):
        bpms, times = _orig_get_tempo_changes(self)
        # Cast only when needed to avoid extra copies.
        if not hasattr(bpms, "dtype"):
            bpms = _np.asarray(bpms)
        if not hasattr(times, "dtype"):
            times = _np.asarray(times)
        return bpms, times

    _safe_get_tempo_changes._composer2_patched = True  # type: ignore[attr-defined]
    _pm.PrettyMIDI.get_tempo_changes = _safe_get_tempo_changes  # type: ignore[assignment]

    def _safe_write(self, filename):
        try:
            filename = _os.fspath(filename)
        except TypeError:
            pass
        except Exception:
            pass
        return _orig_write(self, filename)

    _pm.PrettyMIDI.write = _safe_write  # type: ignore[assignment]
