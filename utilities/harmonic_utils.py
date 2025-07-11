from __future__ import annotations

from collections.abc import Sequence

from music21 import note as m21note
from music21 import pitch

_BASE_MIDIS = [40, 45, 50, 55, 59, 64]

_NATURAL_INTERVALS = {5: 31, 7: 19, 12: 12}


def choose_harmonic(
    base_pitch: pitch.Pitch,
    tuning_offsets: Sequence[int] | None,
    chord_pitches: Sequence[pitch.Pitch] | None,
    max_fret: int = 19,
    base_midis: Sequence[int] | None = None,
) -> tuple[pitch.Pitch, dict] | None:
    """Return a harmonic candidate for *base_pitch*.

    Natural nodes at the 5th, 7th and 12th fret are tried first.  ``tuning_offsets``
    and ``base_midis`` describe the tuning and open-string pitches.  If none fit
    within ``max_fret`` an artificial octave harmonic is attempted.  ``meta``
    describes the chosen string, type and sounding pitch.
    """
    try:
        base_midi = int(round(base_pitch.midi))
    except Exception:
        return None

    tuning_offsets = list(tuning_offsets or [])
    base_midis = list(base_midis or _BASE_MIDIS)
    open_midis: list[int] = []
    for i, m in enumerate(base_midis):
        offset = tuning_offsets[i] if i < len(tuning_offsets) else 0
        open_midis.append(m + offset)

    for idx, open_m in enumerate(open_midis):
        fret = base_midi - open_m
        if not 0 <= fret <= max_fret:
            continue
        for nat_fret in (5, 7, 12):
            touch = fret + nat_fret
            if touch <= max_fret:
                sounding = pitch.Pitch()
                sounding.midi = base_midi + _NATURAL_INTERVALS[nat_fret]
                return sounding, {
                    "type": "natural",
                    "string_idx": idx,
                    "touch_fret": touch,
                    "sounding_pitch": int(sounding.midi),
                    "open_midi": int(open_m),
                }
        touch = fret + 12
        if touch <= max_fret:
            sounding = pitch.Pitch()
            sounding.midi = base_midi + 12
            return sounding, {
                "type": "artificial",
                "string_idx": idx,
                "touch_fret": touch,
                "sounding_pitch": int(sounding.midi),
                "open_midi": int(open_m),
            }
    return None


def apply_harmonic_notation(
    n: m21note.NotRest, meta: dict
) -> None:
    """Attach MusicXML harmonic technical indication to *n* using *meta*.

    ``meta`` must contain ``string_idx`` and ``touch_fret`` as produced by
    :func:`choose_harmonic`.
    """
    try:
        from music21 import articulations

        if not hasattr(n, "notations") or n.notations is None:
            n.notations = m21note.Notations()
        harm = articulations.StringHarmonic()
        harm.harmonicType = "artificial" if meta.get("type") != "natural" else "natural"
        harm.pitchType = "touching"
        n.notations.append(harm)
        StringCls = getattr(articulations, "StringIndication", None)
        FretCls = getattr(articulations, "FretIndication", None)
        if StringCls:
            n.notations.append(StringCls(number=int(meta["string_idx"]) + 1))
        if FretCls:
            n.notations.append(FretCls(number=int(meta["touch_fret"])))
        setattr(n, "string", meta.get("string_idx"))
        setattr(n, "fret", meta.get("touch_fret"))
        n.harmonic_meta = meta
    except Exception:
        pass
