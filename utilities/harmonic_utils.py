from __future__ import annotations

import random
from collections.abc import Sequence
from collections.abc import Sequence as SeqType

from music21 import articulations, pitch
from music21 import note as m21note

_BASE_MIDIS: list[int] = [40, 45, 50, 55, 59, 64]

_NATURAL_INTERVALS: dict[int, int] = {5: 31, 7: 19, 12: 12}


def choose_harmonic(
    base_pitch: pitch.Pitch,
    tuning_offsets: Sequence[int] | None,
    chord_pitches: Sequence[pitch.Pitch] | None,
    max_fret: int = 19,
    open_string_midis: Sequence[int] | None = None,
) -> tuple[pitch.Pitch, dict] | None:
    """Return a harmonic candidate for *base_pitch*.

    Natural nodes at the 5th, 7th and 12th fret are tried first.  ``tuning_offsets``
    and ``open_string_midis`` describe the tuning and open-string pitches.  If none fit
    within ``max_fret`` an artificial octave harmonic is attempted.  ``meta``
    describes the chosen string, type and sounding pitch.
    """
    try:
        base_midi = int(round(base_pitch.midi))
    except Exception:
        return None

    tuning_offsets = list(tuning_offsets or [])
    base_midis = list(open_string_midis or _BASE_MIDIS)
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


def apply_harmonic_to_pitch(
    p: pitch.Pitch,
    *,
    chord_pitches: SeqType[pitch.Pitch],
    tuning_offsets: SeqType[int] | None,
    base_midis: SeqType[int] | None,
    max_fret: int,
    allowed_types: SeqType[str],
    rng: random.Random,
    prob: float,
    volume_factor: float,
    gain_db: float | None = None,
) -> tuple[pitch.Pitch, list[articulations.Articulation], float, dict | None]:
    """Maybe convert *p* into a harmonic pitch.

    Returns the possibly modified pitch, articulation list, velocity factor and
    harmonic metadata. If harmonics are not applied the original pitch is
    returned with an empty articulation list and factor 1.0.
    """
    from music21 import articulations

    if rng.random() >= prob:
        return p, [], 1.0, None

    result = choose_harmonic(
        p,
        tuning_offsets=tuning_offsets,
        chord_pitches=list(chord_pitches),
        max_fret=max_fret,
        open_string_midis=list(base_midis) if base_midis is not None else None,
    )
    if result is None:
        return p, [], 1.0, None
    new_pitch, meta = result
    if meta.get("type") not in allowed_types:
        if "artificial" in allowed_types:
            base_midi = int(round(p.midi))
            open_midis = [
                m + (tuning_offsets[i] if tuning_offsets and i < len(tuning_offsets) else 0)
                for i, m in enumerate(base_midis or _BASE_MIDIS)
            ]
            for idx, open_m in enumerate(open_midis):
                fret = base_midi - open_m
                if 0 <= fret <= max_fret and fret + 12 <= max_fret:
                    new_pitch = pitch.Pitch()
                    new_pitch.midi = base_midi + 12
                    meta = {
                        "type": "artificial",
                        "string_idx": idx,
                        "touch_fret": fret + 12,
                        "sounding_pitch": int(new_pitch.midi),
                    }
                    break
            else:
                return p, [], 1.0, None
        else:
            return p, [], 1.0, None

    arts: list[articulations.Articulation] = [articulations.Harmonic()]
    typ = meta.get("type")
    if typ == "natural" and hasattr(articulations, "NaturalHarmonic"):
        arts.append(articulations.NaturalHarmonic())
    elif typ != "natural" and hasattr(articulations, "ArtificialHarmonic"):
        arts.append(articulations.ArtificialHarmonic())
    if typ != "natural" and hasattr(articulations, "TouchingPitch"):
        arts.append(articulations.TouchingPitch())

    factor = 10 ** (gain_db / 20) if gain_db is not None else volume_factor
    if typ == "artificial":
        factor *= 0.8
    return new_pitch, arts, factor, meta
