from __future__ import annotations

from music21 import note, stream

_DEFAULT_PROFILE: dict[str, int] = {"finger": 28, "slap": 25, "mute": 29}


def add_key_switches(part: stream.Part, profile: dict[str, int] | None = None) -> stream.Part:
    """Insert articulation key switch notes into ``part``.

    The first key switch is inserted two beats before the first note.
    ``profile`` maps articulation names to MIDI pitches. Unspecified
    values fall back to ``_DEFAULT_PROFILE``.
    """
    mapping = _DEFAULT_PROFILE.copy()
    if profile:
        mapping.update(profile)

    # use "finger" articulation as default initial switch
    ks_pitch = mapping.get("finger", _DEFAULT_PROFILE["finger"])
    ks = note.Note(ks_pitch, quarterLength=0.0)
    part.insert(-2.0, ks)  # two beats before start
    return part


__all__ = ["add_key_switches"]


def add_portamento(part: stream.Part, slide_events: list[dict]) -> stream.Part:
    """Insert portamento CC events for each slide.

    Parameters
    ----------
    part : stream.Part
        Target part to annotate.
    slide_events : list[dict]
        Sequence of slide descriptors with ``start`` and ``end`` MIDI pitches,
        ``offset`` in beats and ``duration`` in beats.

    Returns
    -------
    stream.Part
        The modified part with ``extra_cc`` entries.
    """

    extra = getattr(part, "extra_cc", [])
    for ev in slide_events:
        off = float(ev.get("offset", 0.0))
        dur = float(ev.get("duration", 0.5))
        time_val = max(0, min(127, int(dur * 127)))
        extra.append({"time": off, "cc": 5, "val": time_val})
        extra.append({"time": off, "cc": 84, "val": 127})
    part.extra_cc = extra
    return part


__all__.append("add_portamento")
