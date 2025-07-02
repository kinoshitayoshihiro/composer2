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
