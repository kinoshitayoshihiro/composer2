"""Compat shim for pretty_midi.

Some environments (old forks / stubs) make PrettyMIDI.get_tempo_changes()
return plain Python lists, which breaks pretty_midi.PrettyMIDI.get_end_time()
because it blindly calls `.tolist()`.

By importing this module once at startup, we monkey-patch the method so it
*always* returns numpy.ndarray while exposing the legacy (bpms, times)
contract expected elsewhere in the project, only swapping when legacy forks
reversed the order.
"""

from __future__ import annotations

import os as _os
import numpy as _np

try:  # pragma: no cover - exercised in pretty_midi environments
    import pretty_midi as _pm
except ImportError:  # pragma: no cover - allows import without pretty_midi
    _pm = None
else:
    def _ensure_array(seq):
        if hasattr(seq, "dtype"):
            return seq
        return _np.asarray(seq, dtype=float)

    def _times_score(arr):
        score = 0
        if getattr(arr, "ndim", 1) == 1:
            score += 1
        size = getattr(arr, "size", len(arr))
        if size == 0:
            return score
        try:
            first = float(arr[0])
        except (TypeError, ValueError):
            return score
        if abs(first) <= 1e-3:
            score += 2
        try:
            diffs = _np.diff(arr)
        except (TypeError, ValueError):
            return score
        if diffs.size == 0 or _np.all(diffs >= -1e-9):
            score += 2
        if size and float(arr[-1]) >= first:
            score += 1
        return score

    def _bpms_score(arr):
        score = 0
        if getattr(arr, "ndim", 1) == 1:
            score += 1
        size = getattr(arr, "size", len(arr))
        if size == 0:
            return score
        if hasattr(arr, "dtype") and arr.dtype.kind in "fiu" and _np.isfinite(arr).all():
            arr_view = arr
        else:
            try:
                arr_view = _np.asarray(arr, dtype=float)
            except (TypeError, ValueError):
                return score
        if not _np.isfinite(arr_view).all():
            return score
        if _np.all(arr_view > 0):
            score += 2
            max_val = float(arr_view.max())
            if max_val <= 600:
                score += 2
            elif max_val <= 1000:
                score += 1
        return score

    # If someone else already patched it, don't stack.
    if not getattr(_pm.PrettyMIDI.get_tempo_changes, "_composer2_patched", False):
        _orig_get_tempo_changes = _pm.PrettyMIDI.get_tempo_changes
        _orig_write = _pm.PrettyMIDI.write

        def _safe_get_tempo_changes(self):
            first, second = _orig_get_tempo_changes(self)
            first_arr = _ensure_array(first)
            second_arr = _ensure_array(second)

            score_first_as_times = _times_score(first_arr) + _bpms_score(second_arr)
            score_second_as_times = _times_score(second_arr) + _bpms_score(first_arr)

            # Tie-break: prefer the official pretty_midi order (times, bpms)
            # internally, then flip back to the legacy (bpms, times) that the
            # rest of the project consumes.
            if score_second_as_times > score_first_as_times:
                times, bpms = second_arr, first_arr
            else:
                times, bpms = first_arr, second_arr
            return bpms, times

        _safe_get_tempo_changes._composer2_patched = True  # type: ignore[attr-defined]
        _pm.PrettyMIDI.get_tempo_changes = _safe_get_tempo_changes  # type: ignore[assignment]

        # PathLike を常に文字列へ（古い mido が Path を file-like と誤解）
        def _safe_write(self, filename):
            try:
                filename = _os.fspath(filename)
            except Exception:
                # If it isn't PathLike or fspath fails, pass through as-is.
                pass
            return _orig_write(self, filename)

        _pm.PrettyMIDI.write = _safe_write  # type: ignore[assignment]
