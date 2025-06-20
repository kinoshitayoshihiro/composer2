from typing import TypedDict, List

# Mapping from simple DSL tokens to drum instrument labels

TOKEN_MAP: dict[str, str] = {
    "T1": "tom1",
    "T2": "tom2",
    "T3": "tom3",
    "K": "kick",
    "SN": "snare",
    "CRASH": "crash",
}

RUN_TOKENS = {
    "RUNUP": "up",
    "RUNDOWN": "down",
    "RUN↑": "up",
    "RUN↓": "down",
}


def _parse_run(tok: str) -> tuple[str, int] | None:
    key = tok.upper()
    for base, direction in RUN_TOKENS.items():
        if key.startswith(base):
            rest = key[len(base) :]
            count = 3
            if rest.startswith("X") and rest[1:].isdigit():
                count = int(rest[1:])
            return direction, count
    return None


class DrumEvent(TypedDict):
    instrument: str
    offset: float
    duration: float
    velocity_factor: float


def _expand_run(
    direction: str, length_beats: float, vel_factor: float, count: int = 3
) -> List[DrumEvent]:
    """Return tom run events.

    Parameters
    ----------
    direction : str
        Either ``"up"`` or ``"down"``.
    length_beats : float
        Total run length in beats.
    vel_factor : float
        Base velocity scaling factor.

    Returns
    -------
    list[DrumEvent]
        Expanded events evenly spaced within ``length_beats``.
    """
    base = ["tom1", "tom2", "tom3"] if direction == "up" else ["tom3", "tom2", "tom1"]
    drums = [base[i % 3] for i in range(max(1, count))]
    drums.append("crash" if direction == "up" else "kick")
    length = max(float(length_beats), 0.001)
    step = length / len(drums)
    events: List[DrumEvent] = []
    for idx, inst in enumerate(drums):
        scale = (
            (idx + 1) / len(drums)
            if direction == "up"
            else (len(drums) - idx) / len(drums)
        )
        events.append(
            {
                "instrument": inst,
                "offset": idx * step,
                "duration": step if idx < len(drums) - 1 else 0.25,
                "velocity_factor": float(vel_factor) * scale,
            }
        )
    return events


def parse(
    template: str,
    length_beats: float = 1.0,
    velocity_factor: float = 1.0,
) -> List[DrumEvent]:
    """Parse simple drum fill DSL into event dictionaries.

    Parameters
    ----------
    template : str
        Space separated token string (e.g. ``"T1 T2 RUNUP"``).
        ``RUNUP``/``RUNDOWN`` (or ``RUN↑``/``RUN↓``) expand to an ascending
        or descending tom run. ``RUNUPx6`` specifies six tom hits before the
        final crash (kick for ``RUNDOWNxN``).
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

    # determine total event count considering RUN tokens
    total = 0
    parsed_tokens: list[tuple[str, int] | str] = []
    for tok in tokens:
        run_info = _parse_run(tok)
        if run_info:
            direction, count = run_info
            total += count + 1
            parsed_tokens.append((direction, count))
        else:
            key = tok.upper()
            if key not in TOKEN_MAP:
                raise KeyError(key)
            total += 1
            parsed_tokens.append(key)

    length = max(float(length_beats), 0.001)
    step = length / total
    events: List[DrumEvent] = []
    idx = 0
    for token in parsed_tokens:
        if isinstance(token, tuple):
            direction, count = token
            run_events = _expand_run(
                direction, step * (count + 1), velocity_factor, count
            )
            start_offset = idx * step
            for ev in run_events:
                ev["offset"] += start_offset
                events.append(ev)
                idx += 1
            continue
        key = token

        offset = idx * step
        duration = step if idx < total - 1 else 0.25
        events.append(
            {
                "instrument": TOKEN_MAP[key],
                "offset": offset,
                "duration": duration,
                "velocity_factor": float(velocity_factor),
            }
        )
        idx += 1

    return events
