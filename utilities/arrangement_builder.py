from __future__ import annotations

from typing import Iterable
from copy import deepcopy
import io

from music21 import stream, expressions, instrument
from music21.midi import translate
import pretty_midi


def _clone(elem: object) -> object:
    """Return a safe clone of *elem*."""
    try:
        return elem.clone()
    except Exception:
        return deepcopy(elem)


SectionParts = tuple[str, float, dict[str, stream.Part]]


def build_arrangement(sections: Iterable[SectionParts]) -> tuple[stream.Score, list[str]]:
    """Combine per-section parts into one part per generator.

    Parameters
    ----------
    sections:
        Iterable of ``(section_name, offset_q, parts)`` tuples. ``parts`` maps a
        generator name to a ``music21`` ``Part`` representing that section.

    Returns
    -------
    tuple[stream.Score, list[str]]
        The merged score and ordered list of generator names.
    """
    combined: dict[str, stream.Part] = {}
    generator_names: list[str] = []

    for section_name, offset_q, parts in sections:
        for gen_name, part in parts.items():
            if gen_name not in combined:
                dest = stream.Part(id=gen_name)
                inst = part.getInstrument(returnDefault=False)
                if inst is None:
                    try:
                        inst = instrument.fromString(gen_name)
                    except Exception:
                        inst = instrument.Instrument()
                        inst.partName = gen_name
                dest.insert(0.0, inst)
                combined[gen_name] = dest
                generator_names.append(gen_name)
            else:
                dest = combined[gen_name]

            dest.insert(offset_q, expressions.TextExpression(section_name))
            for elem in part:
                if isinstance(elem, instrument.Instrument):
                    continue
                dest.insert(offset_q + elem.offset, _clone(elem))

    score = stream.Score()
    for name in generator_names:
        score.insert(0, combined[name])
    return score, generator_names


def score_to_pretty_midi(score: stream.Score) -> pretty_midi.PrettyMIDI:
    """Convert a ``music21`` score to ``pretty_midi``."""
    mf = translate.streamToMidiFile(score)
    data = mf.writestr()
    return pretty_midi.PrettyMIDI(io.BytesIO(data))
