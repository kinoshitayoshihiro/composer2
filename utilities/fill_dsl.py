from typing import TypedDict, List

TOKEN_MAP: dict[str, str] = {
    "T1": "tom1",
    "T2": "tom2",
    "T3": "tom3",
    "K": "kick",
    "SN": "snare",
    "CRASH": "crash",
}


class DrumEvent(TypedDict):
    instrument: str
    offset: float
    duration: float
    velocity_factor: float

def parse(
    template: str,
    length_beats: float = 1.0,
    velocity_factor: float = 1.0,
) -> List[DrumEvent]:
    """Parse simple drum fill DSL into event dictionaries.

    Parameters
    ----------
    template : str
        Space separated token string (e.g. "T1 T2 T3 CRASH").
    length_beats : float, optional
        Total length in beats. Offsets are distributed evenly across this span.
    velocity_factor : float, optional
        Velocity multiplier applied to all events.

    Returns
    -------
    List[DrumEvent]
        Sorted list of event dictionaries ready for music21 insertion.

    Raises
    ------
    KeyError
        If an unknown token is encountered.
    """
    tokens = [tok for tok in (template or "").split() if tok]
    if not tokens:
        return []
    length = max(float(length_beats), 0.001)
    step = length / len(tokens)
    events: List[DrumEvent] = []
    for idx, token in enumerate(tokens):
        key = token.upper()
        if key not in TOKEN_MAP:
            raise KeyError(key)
        offset = idx * step
        duration = step if idx < len(tokens) - 1 else 0.25
        events.append({
            "instrument": TOKEN_MAP[key],
            "offset": offset,
            "duration": duration,
            "velocity_factor": float(velocity_factor),
        })
    return events
