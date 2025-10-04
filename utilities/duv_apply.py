"""Lightweight helpers to apply DUV humanisation inside the codebase.

This module intentionally wraps the public functions exposed by
``scripts.duv_infer`` so that runtime code can reuse the CLI implementation
without duplicating logic.  The only additional behaviour layered on top is a
conservative safeguard for key-switch / trigger notes so downstream UJAM,
SynthV, and VOCALOID workflows remain compatible.
"""

from __future__ import annotations

import importlib
import logging
import re
from typing import Any, Callable, Optional, cast

import pretty_midi as pm  # type: ignore[import-untyped]

try:  # ``scripts.duv_infer`` lives alongside the CLI entry-point
    _duv_module = importlib.import_module("scripts.duv_infer")
except ImportError as exc:  # pragma: no cover - defensive import guard
    raise RuntimeError("scripts.duv_infer module is required for DUV application") from exc

ComputeBeatsFn = Callable[[pm.PrettyMIDI], tuple[Any, Any]]
ProcessInstrumentFn = Callable[..., None]

DUVModel: Any = getattr(_duv_module, "DUVModel")
Scaler: Any = getattr(_duv_module, "Scaler")
compute_beats = cast(ComputeBeatsFn, getattr(_duv_module, "compute_beats"))
process_instrument = cast(ProcessInstrumentFn, getattr(_duv_module, "process_instrument"))

LOGGER = logging.getLogger(__name__)

# Notes with pitches at or below this value are treated as potential
# key-switches.  UJAM/SynthV/VOCALOID products typically dedicate the
# sub-C1 octave (<=24) to articulation switches; real musical content rarely
# lives there outside of contrabass extensions, so this default keeps them
# untouched while remaining conservative.
_DEFAULT_PROTECT_PITCH_BELOW = 24
# Extremely short events (e.g. trigger taps) should retain their original gate.
_DEFAULT_PROTECT_DURATION_SEC = 0.06


def _compile(pattern: Optional[str]) -> Optional[re.Pattern[str]]:
    if pattern:
        try:
            return re.compile(pattern)
        except re.error as exc:  # pragma: no cover - configuration error
            LOGGER.warning("Invalid regex '%s': %s", pattern, exc)
    return None


def _should_process(
    name: str,
    *,
    include: Optional[re.Pattern[str]],
    exclude: Optional[re.Pattern[str]],
) -> bool:
    if include is not None and not include.search(name):
        return False
    if exclude is not None and exclude.search(name):
        return False
    return True


def _snapshot_notes(inst: Any) -> list[tuple[int, float, float, int]]:
    """Return (pitch, start, end, velocity) tuples for all notes."""
    return [(int(n.pitch), float(n.start), float(n.end), int(n.velocity)) for n in inst.notes]


def _restore_protected_notes(
    inst: Any,
    original: list[tuple[int, float, float, int]],
    *,
    protect_pitch_below: int,
    protect_duration_sec: float,
) -> None:
    for note, (pitch, start, end, velocity) in zip(inst.notes, original):
        if pitch <= protect_pitch_below:
            note.velocity = velocity
            note.start = start
            note.end = end
            continue
        if (end - start) <= protect_duration_sec:
            note.velocity = velocity
            note.start = start
            note.end = end


def apply_duv_to_pretty_midi(
    midi: pm.PrettyMIDI,
    *,
    model_path: str,
    scaler_path: Optional[str] = None,
    mode: str = "absolute",
    intensity: float = 1.0,
    include_regex: Optional[str] = None,
    exclude_regex: Optional[str] = None,
    include_drums: bool = False,
    protect_pitch_below: int = _DEFAULT_PROTECT_PITCH_BELOW,
    protect_duration_sec: float = _DEFAULT_PROTECT_DURATION_SEC,
) -> pm.PrettyMIDI:
    """Apply a DUV model to ``midi`` and return the modified object.

    Parameters mirror the CLI flags in :mod:`scripts.duv_infer` so the same
    checkpoints and scaler artefacts can be reused.  The function mutates the
    provided ``PrettyMIDI`` instance in-place and also returns it for
    convenience.

    Notes
    -----
    * Key-switch or trigger notes (pitches ``<= protect_pitch_below`` or gate
      shorter than ``protect_duration_sec`` seconds) are restored to their
      pre-DUV values to keep downstream instrument mappings intact.
    * ``mode`` must be either ``"absolute"`` or ``"delta"``; the helper defers
      to :func:`scripts.duv_infer.process_instrument` for the heavy lifting.
    """

    if intensity <= 0.0:
        LOGGER.debug("DUV intensity %.3f clamps to 0 — skipping", intensity)
        return midi

    include_re = _compile(include_regex)
    exclude_re = _compile(exclude_regex)

    beat_times, beat_nums = compute_beats(midi)
    model = DUVModel.load(model_path)
    scaler = Scaler.from_json(scaler_path) if scaler_path else None

    pm_obj = cast(Any, midi)

    for inst in pm_obj.instruments:
        if inst.is_drum and not include_drums:
            continue
        name = inst.name or ""
        if not _should_process(name, include=include_re, exclude=exclude_re):
            continue
        if not inst.notes:
            continue

        original = _snapshot_notes(inst)
        try:
            process_instrument(
                inst,
                model,
                scaler,
                mode=mode,
                intensity=float(intensity),
                beat_times=beat_times,
                beat_nums=beat_nums,
                dry_run_rows=None,  # type: ignore[arg-type]
            )
        except (
            ValueError,
            RuntimeError,
            KeyError,
            IndexError,
            TypeError,
        ) as exc:  # pragma: no cover - inference runtime errors
            # Intentionally broad so any inference glitch falls back to the
            # original notes.
            LOGGER.warning("DUV inference failed for '%s': %s", name, exc)
            inst.notes[:] = []
            for pitch, start, end, velocity in original:
                inst.notes.append(
                    pm.Note(
                        velocity=velocity,
                        pitch=pitch,
                        start=start,
                        end=end,
                    )
                )
            continue

        _restore_protected_notes(
            inst,
            original,
            protect_pitch_below=protect_pitch_below,
            protect_duration_sec=protect_duration_sec,
        )
        inst.notes[:] = sorted(
            inst.notes,
            key=lambda n: (
                float(getattr(n, "start", 0.0)),
                int(getattr(n, "pitch", 0)),
            ),
        )

    return midi


__all__ = ["apply_duv_to_pretty_midi"]
