from typing import Any, cast

from utilities.groove_sampler_ngram import Event


def make_event(
    *,
    instrument: str,
    offset: float,
    duration: float = 0.25,
    velocity: int = 100,
    **extra: Any,
) -> Event:
    """Create an ``Event`` with defaults and allow extra keys."""
    data: dict[str, Any] = {
        "instrument": instrument,
        "offset": float(offset),
        "duration": float(duration),
        "velocity": int(velocity),
    }
    data.update(extra)
    return cast(Event, data)
