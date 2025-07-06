from __future__ import annotations

"""CounterLineGenerator creates a simple counter melody above a given melody."""

from music21 import interval, note, stream


class CounterLineGenerator:
    """Generate counter line based on an input melody."""

    def generate(self, melody: stream.Part, scale: str = "major") -> stream.Part:
        """Return a counter line part.

        Parameters
        ----------
        melody : :class:`music21.stream.Part`
            Source melody part.
        scale : str, optional
            Currently unused scale name. Included for future extension.
        """
        counter = stream.Part(id="CounterLine")
        notes = list(melody.flat.getElementsByClass(note.Note))
        if not notes:
            return counter

        prev = None
        for src in notes:
            tgt_pitch = interval.Interval("-P5").transposePitch(src.pitch)
            tgt_pitch = interval.Interval("P8").transposePitch(tgt_pitch)
            tgt = note.Note(tgt_pitch, quarterLength=src.quarterLength)
            counter.insert(src.offset, tgt)

            if prev is not None:
                semitones = tgt.pitch.midi - prev.pitch.midi
                if abs(semitones) > 4:
                    step = 2 if semitones > 0 else -2
                    mid_off = (prev.offset + src.offset) / 2
                    passing = note.Note(prev.pitch.transpose(step), quarterLength=0.5)
                    counter.insert(mid_off, passing)
            prev = tgt

        return counter

__all__ = ["CounterLineGenerator"]
