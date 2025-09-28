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
from typing import Any, cast

import numpy as _np

_NO_TEMPO_MARKER = "__composer2_no_tempo__"

try:  # pragma: no cover - exercised in pretty_midi environments
    import pretty_midi as _pm  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - allows import without pretty_midi
    _pm = None
else:

    def _ensure_array(seq: Any) -> Any:
        return _np.asarray(seq, dtype=float)

    def _times_score(arr: Any) -> int:
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

    def _bpms_score(arr: Any) -> int:
        score = 0
        if getattr(arr, "ndim", 1) == 1:
            score += 1
        size = getattr(arr, "size", len(arr))
        if size == 0:
            return score
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
    if not getattr(
        _pm.PrettyMIDI.get_tempo_changes,
        "_composer2_patched",
        False,
    ):
        _orig_get_tempo_changes: Any = getattr(
            _pm.PrettyMIDI,
            "get_tempo_changes",
        )
        _orig_write: Any = getattr(_pm.PrettyMIDI, "write")

        def _tick_scales_to_arrays(
            self: Any,
        ) -> Any:
            tick_scales = getattr(self, "_tick_scales", None)
            if not tick_scales:
                return None

            try:
                resolution = float(getattr(self, "resolution"))
            except (AttributeError, TypeError, ValueError):
                return None

            times: list[float] = []
            bpms: list[float] = []
            last_tick = None
            prev_scale = None
            current_time = 0.0

            for idx, pair in enumerate(tick_scales):
                try:
                    tick, scale = pair
                except (TypeError, ValueError):
                    return None

                try:
                    tick = int(tick)
                    scale = float(scale)
                except (TypeError, ValueError):
                    return None

                if idx == 0:
                    current_time = max(0.0, tick * scale)
                    times.append(0.0 if tick == 0 else current_time)
                else:
                    if last_tick is None or prev_scale is None:
                        return None
                    delta = tick - last_tick
                    if delta < 0:
                        return None
                    current_time += delta * prev_scale
                    times.append(current_time)

                if scale <= 0 or resolution <= 0:
                    return None
                bpm = 60.0 / (scale * resolution)
                if not _np.isfinite(bpm) or bpm <= 0:
                    return None
                bpms.append(bpm)

                last_tick = tick
                prev_scale = scale

            if not times:
                return None
            return _ensure_array(times), _ensure_array(bpms)

        def _safe_get_tempo_changes(
            self: Any,
        ) -> Any:
            first, second = _orig_get_tempo_changes(self)
            first_arr = _ensure_array(first)
            second_arr = _ensure_array(second)

            score_first_as_times = _times_score(first_arr) + _bpms_score(second_arr)
            score_second_as_times = _times_score(second_arr) + _bpms_score(first_arr)

            # Tie-break: prefer the official pretty_midi order (times, bpms)
            if score_second_as_times > score_first_as_times:
                times, bpms = second_arr, first_arr
            else:
                times, bpms = first_arr, second_arr

            midi_obj = getattr(self, "midi_data", None)
            if midi_obj is not None:
                has_tempo_meta = any(
                    getattr(msg, "type", "") == "set_tempo"
                    for track in getattr(midi_obj, "tracks", [])
                    for msg in track
                )
            else:
                has_tempo_meta = False

            marker_present = any(
                getattr(evt, "text", "") == _NO_TEMPO_MARKER
                for evt in getattr(self, "text_events", [])
            )
            if marker_present:
                self.text_events = [
                    evt for evt in self.text_events if getattr(evt, "text", "") != _NO_TEMPO_MARKER
                ]

            if not has_tempo_meta:
                arrays = _tick_scales_to_arrays(self)
                if arrays is not None:
                    return arrays
                empty = _np.asarray([], dtype=float)
                return empty, empty
            return times, bpms

        setattr(_safe_get_tempo_changes, "_composer2_patched", True)
        setattr(_pm.PrettyMIDI, "get_tempo_changes", _safe_get_tempo_changes)

        # PathLike を常に文字列へ（古い mido が Path を file-like と誤解）
        def _safe_write(self: Any, filename: Any):
            try:
                filename = cast(str | bytes, _os.fspath(filename))
            except TypeError:
                # If it isn't PathLike or fspath fails, pass through as-is.
                pass
            return _orig_write(self, filename)

        _pm.PrettyMIDI.write = _safe_write  # type: ignore[assignment]
