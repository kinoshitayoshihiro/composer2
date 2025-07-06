from __future__ import annotations

import copy
from typing import List

from music21 import note


class VoicingDensityEngine:
    """Simple note density scaler based on intensity."""

    def scale_density(self, notes: List[note.NotRest], intensity: str) -> List[note.NotRest]:
        """Return a new list of notes scaled by *intensity*.

        Parameters
        ----------
        notes:
            List of ``note.Note`` or ``chord.Chord`` objects with ``offset`` set.
        intensity:
            One of ``"low"``, ``"medium"`` or ``"high"``.
        """
        if intensity not in {"low", "medium", "high"}:
            return list(notes)

        notes_sorted = sorted(notes, key=lambda n: float(n.offset))
        if intensity == "low":
            target_len = max(1, int(round(len(notes_sorted) * 0.5)))
            to_remove = len(notes_sorted) - target_len
            if to_remove <= 0:
                return list(notes_sorted)
            even_indices = [
                i
                for i, n in enumerate(notes_sorted)
                if int(round(float(n.offset))) % 2 == 1
            ]
            remove_order = even_indices + [i for i in range(len(notes_sorted)) if i not in even_indices]
            remove_set = set(remove_order[:to_remove])
            return [n for i, n in enumerate(notes_sorted) if i not in remove_set]

        if intensity == "medium":
            # Near-neutral density; keep original
            return list(notes_sorted)

        if intensity == "high":
            new_notes: List[note.NotRest] = []
            for n in notes_sorted:
                new_notes.append(n)
                try:
                    anticip = copy.deepcopy(n)
                    anticip.offset = float(n.offset) - 0.5
                    if anticip.offset >= 0:
                        new_notes.append(anticip)
                except Exception:
                    continue
            new_notes.sort(key=lambda n: float(n.offset))
            return new_notes

        return list(notes_sorted)
