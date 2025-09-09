"""Simple tokenizer for sax events used in tests.

The implementation intentionally keeps the feature set tiny.  It only
understands note numbers ``0-127`` and three articulation markers.  All
other event information is stored dynamically so that a round trip via
``encode`` and ``decode`` preserves the original dictionaries.
"""

from __future__ import annotations

import warnings


class SaxTokenizer:
    """Tokenizer for saxophone events."""

    pad_id = 0

    def __init__(self) -> None:
        # Static vocabulary containing note tokens and articulation markers.
        self.vocab: dict[str, int] = {"<pad>": self.pad_id}
        self.note_to_id: dict[int, int] = {}
        for i in range(128):
            tok_id = len(self.vocab)
            self.vocab[f"NOTE_{i}"] = tok_id
            self.note_to_id[i] = tok_id

        for name in ("<SLIDE_UP>", "<SLIDE_DOWN>", "<ALT_HIT>"):
            self.vocab[name] = len(self.vocab)

        # Runtime mappings used to keep full events reversible.
        self.event_to_id: dict[tuple[tuple[str, object], ...], int] = {}
        self.id_to_event: dict[int, dict[str, object]] = {}
        self.next_id = len(self.vocab)

    def encode(self, events: list[dict[str, object]]) -> list[int]:
        ids: list[int] = []
        for ev in events:
            note = ev.get("note")
            if not isinstance(note, int) or not (0 <= note < 128):
                warnings.warn("unk token rate - unknown note")

            artic = ev.get("artic")
            if artic not in {None, "slide_up", "slide_down", "alt_hit"}:
                warnings.warn("unk token rate - unknown articulation")

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
                # Basic note token without any other info
                note_num = [n for n, tok in self.note_to_id.items() if tok == i][0]
                events.append({"note": note_num})
            else:
                warnings.warn("unk token rate - unknown id")
                events.append({})
        return events




__all__ = ["SaxTokenizer"]
