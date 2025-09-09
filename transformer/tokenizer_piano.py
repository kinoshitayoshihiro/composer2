"""Tokenizer for simple piano events.

The aim is not to provide a production ready tokenizer but merely a
deterministic mapping so that unit tests can round-trip event structures.
Unknown values trigger warnings and are otherwise ignored.
"""

from __future__ import annotations

import warnings


class PianoTokenizer:
    """Tokenizer used in tests."""

    pad_id = 0

    def __init__(self) -> None:
        # Static vocabulary of note tokens.
        self.vocab: dict[str, int] = {"<pad>": self.pad_id}
        self.note_to_id: dict[int, int] = {}
        for i in range(128):
            tok = len(self.vocab)
            self.vocab[f"NOTE_{i}"] = tok
            self.note_to_id[i] = tok

        # Runtime mappings for full events.
        self.event_to_id: dict[tuple[tuple[str, object], ...], int] = {}
        self.id_to_event: dict[int, dict[str, object]] = {}
        self.next_id = len(self.vocab)

    def encode(self, events: list[dict[str, object]]) -> list[int]:
        ids: list[int] = []
        for ev in events:
            note = ev.get("note")
            if not isinstance(note, int) or not (0 <= note < 128):
                warnings.warn("unk token rate - unknown note")

            hand = ev.get("hand")
            if hand not in {None, "lh", "rh"}:
                warnings.warn("unk token rate - unknown hand")

            key = tuple(sorted(ev.items()))
            if key not in self.event_to_id:
                self.event_to_id[key] = self.next_id
                self.id_to_event[self.next_id] = dict(ev)
                self.next_id += 1
            ids.append(self.event_to_id[key])
        return ids

    def decode(self, ids: list[int]) -> list[dict[str, object]]:
        events: list[dict[str, object]] = []
        for i in ids:
            if i in self.id_to_event:
                events.append(dict(self.id_to_event[i]))
            elif i in self.note_to_id.values():
                note_num = [n for n, tok in self.note_to_id.items() if tok == i][0]
                events.append({"note": note_num})
            else:
                warnings.warn("unk token rate - unknown id")
                events.append({})
        return events




__all__ = ["PianoTokenizer"]
