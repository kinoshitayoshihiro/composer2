from __future__ import annotations

from typing import Dict, Optional

import pretty_midi

try:
    from utilities import humanizer  # apply(), load_profiles() 等
except Exception:  # pragma: no cover - optional dependency
    humanizer = None  # type: ignore

try:
    from tools.ujam_bridge import utils as ujam_utils
except Exception:  # pragma: no cover - optional dependency
    ujam_utils = None  # type: ignore


def apply_humanize_to_instrument(
    inst: pretty_midi.Instrument,
    tempo: float,
    *,
    profile: str = "default_subtle",
    overrides: Optional[Dict] = None,
    quantize: Optional[Dict] = None,
    groove: Optional[Dict] = None,
    late_humanize_ms: int = 0,
) -> None:
    """Apply quantize/groove/humanize pipeline to ``inst``.

    The processing order is quantize → groove → humanize → late humanize.
    Each stage is optional and silently skipped when the corresponding
    dependency is unavailable.
    """

    if quantize and ujam_utils is not None:
        grid = float(quantize.get("grid", 0.25))
        swing = float(quantize.get("swing", 0.0))
        try:
            ujam_utils.quantize(inst, grid=grid, swing=swing, tempo=tempo)
        except Exception:
            pass

    if groove and ujam_utils is not None:
        name = str(groove.get("name", ""))
        if name:
            try:
                ujam_utils.apply_groove_profile(inst, name=name, tempo=tempo)
            except Exception:
                pass

    if humanizer is not None:
        try:
            humanizer.apply(inst, tempo=tempo, profile=profile, **(overrides or {}))
        except Exception:
            pass

    if late_humanize_ms and ujam_utils is not None:
        try:
            ujam_utils.apply_late_humanization(inst, delay_ms=int(late_humanize_ms))
        except Exception:
            pass
