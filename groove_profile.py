from __future__ import annotations
"""
utilities/groove_profile.py
----------------------------------
Extract a micro‑timing "groove profile" from a monophonic (or lightly polyphonic)
MIDI vocal part that has already been decoded into a list‑of‑dicts structure
[{"offset": float, "pitch": str | int, "length": float}, ...].

The profile captures **average timing deviations** for each metrical subdivision
(e.g. 1‑and‑2‑and‑3‑and‑4‑and for 8‑note swing) so that other generators
(Drums, Bass, Guitar …) can shift their events by a similar amount, achieving a
"follow the singer" feel.

Typical usage
-------------
>>> from utilities.groove_profile import extract_groove_profile, apply_groove
>>> profile = extract_groove_profile(vocal_note_list, subdivision="eighth")
>>> new_offset = apply_groove(original_offset, profile)

The resulting profile is **deterministic**, small (JSON‑serialisable) and can be
saved next to the rendered MIDI for later re‑use.
"""
from typing import List, Dict, Literal, Tuple
import math
import statistics

Subdivision = Literal["quarter", "eighth", "sixteenth", "triplet"]

DEFAULT_SUBDIVISION: Subdivision = "eighth"

# -------------------------------- API -------------------------------- #

def extract_groove_profile(
    note_events: List[Dict[str, float | str]],
    *,
    subdivision: Subdivision = DEFAULT_SUBDIVISION,
    time_signature_beats: int = 4,
) -> Dict[str, float]:
    """Analyse *micro‑timing* offsets per subdivision.

    Parameters
    ----------
    note_events : list of dicts
        Each dict must contain at least an "offset" key in *quarter lengths*.
    subdivision : {'quarter','eighth','sixteenth','triplet'}
        Grid resolution used for quantisation reference.
    time_signature_beats : int, default=4
        Needed to wrap the phase back to the start of each bar.

    Returns
    -------
    dict
        Mapping from *grid index* ("0", "0.5" …) to average deviation in
        quarterLength (positive = laid‑back / later, negative = ahead).
    """
    if not note_events:
        return {}

    grid_step = _subdivision_to_step(subdivision)
    grid_points: Dict[float, List[float]] = {}

    for ev in note_events:
        # Ignore rests or malformed items
        try:
            off = float(ev["offset"])
        except (KeyError, TypeError, ValueError):
            continue

        phase_in_bar = off % time_signature_beats  # 0‑beat based
        # Quantised grid point
        grid_idx = round(phase_in_bar / grid_step) * grid_step
        deviation = phase_in_bar - grid_idx
        grid_points.setdefault(grid_idx, []).append(deviation)

    # Aggregate
    profile: Dict[str, float] = {}
    for grid_idx, deviations in grid_points.items():
        # Use *mean*; median could also be interesting
        profile[str(grid_idx)] = statistics.mean(deviations)

    profile["meta_subdivision"] = subdivision  # store for sanity‑check
    profile["meta_grid_step"] = grid_step
    return profile


def apply_groove(original_offset: float, profile: Dict[str, float]) -> float:
    """Shift *original_offset* by the amount stored in *profile*.

    If the exact grid index isn't present (e.g. very sparse vocal), the nearest
    lower index is used as fallback; ultimately defaults to 0 shift.
    """
    grid_step = float(profile.get("meta_grid_step", _subdivision_to_step(DEFAULT_SUBDIVISION)))
    beats = float(profile.get("meta_time_signature_beats", 4))

    phase_in_bar = original_offset % beats
    grid_idx = math.floor(phase_in_bar / grid_step) * grid_step
    shift = profile.get(str(grid_idx), 0.0)
    return original_offset + shift

# --------------------------- helper functions ------------------------- #

def _subdivision_to_step(subdivision: Subdivision) -> float:
    if subdivision == "quarter":
        return 1.0
    elif subdivision == "eighth":
        return 0.5
    elif subdivision == "sixteenth":
        return 0.25
    elif subdivision == "triplet":
        # 8‑note triplet inside a quarter → 1/3 ql
        return 1.0 / 3.0
    else:
        raise ValueError(f"Unsupported subdivision: {subdivision}")

# ----------------------------- CLI / demo ----------------------------- #
if __name__ == "__main__":
    import json, sys
    if len(sys.argv) < 2:
        print("Usage: python -m utilities.groove_profile vocal_note_data.json [eighth]")
        sys.exit(1)

    with open(sys.argv[1], "r", encoding="utf-8") as fp:
        notes = json.load(fp)

    subd = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_SUBDIVISION
    prof = extract_groove_profile(notes, subdivision=subd)  # type: ignore[arg-type]
    print(json.dumps(prof, indent=2, ensure_ascii=False))
