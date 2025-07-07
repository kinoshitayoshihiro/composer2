from __future__ import annotations

"""Tokenizer for saxophone MIDI events."""

import json
import warnings


class SaxTokenizer:
    """Minimal tokenizer for sax events with slide and alt-hit tokens."""

    def __init__(self) -> None:
        specials = [
            "<BAR>",
            "<TS_4_4>",
            "<SLIDE_UP>",
            "<SLIDE_DOWN>",
            "<ALT_HIT>",
            "<REST_d8>",
            "<VELO_80>",
            "<UNK>",
        ]
        pitches = [f"P{i}" for i in range(60, 97)]
        self.vocab = specials + pitches
        self.token_to_id: dict[str, int] = {tok: i for i, tok in enumerate(self.vocab)}
        self.id_to_token: dict[int, str] = {i: tok for i, tok in enumerate(self.vocab)}

    def export_vocab(self, path: str) -> None:
        """Write ``self.vocab`` to ``path`` as JSON list."""
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.vocab, fh)

    def encode(self, events: list[dict[str, object]]) -> list[int]:
        tokens: list[int] = [self.token_to_id["<TS_4_4>"]]
        current_bar = -1
        unk = 0
        for ev in events:
            bar = int(ev.get("bar", 0))
            while bar > current_bar:
                tokens.append(self.token_to_id["<BAR>"])
                current_bar += 1
            pitch_token = f"P{int(ev.get('note', 60))}"
            if pitch_token in self.token_to_id:
                tokens.append(self.token_to_id[pitch_token])
            else:
                tokens.append(self.token_to_id["<UNK>"])
                unk += 1
            tech = str(ev.get("artic", "")).lower()
            if tech == "slide_up":
                tokens.append(self.token_to_id["<SLIDE_UP>"])
            elif tech == "slide_down":
                tokens.append(self.token_to_id["<SLIDE_DOWN>"])
            elif tech == "alt_hit":
                tokens.append(self.token_to_id["<ALT_HIT>"])
            tokens.append(self.token_to_id["<VELO_80>"])
        if unk / max(1, len(tokens)) > 0.01:
            warnings.warn(f"unk token rate {unk/len(tokens):.2%}")
        return tokens

    def decode(self, ids: list[int]) -> list[dict[str, object]]:
        events: list[dict[str, object]] = []
        bar = -1
        i = 0
        while i < len(ids):
            tok = self.id_to_token[ids[i]]
            if tok == "<TS_4_4>":
                bar = -1
                i += 1
                continue
            if tok == "<BAR>":
                bar += 1
                i += 1
                continue
            if tok.startswith("P"):
                pitch = int(tok[1:])
                i += 1
                artic = None
                if i < len(ids):
                    next_tok = self.id_to_token[ids[i]]
                    if next_tok == "<SLIDE_UP>":
                        artic = "slide_up"
                        i += 1
                    elif next_tok == "<SLIDE_DOWN>":
                        artic = "slide_down"
                        i += 1
                    elif next_tok == "<ALT_HIT>":
                        artic = "alt_hit"
                        i += 1
                if i < len(ids) and self.id_to_token[ids[i]] == "<VELO_80>":
                    i += 1
                ev = {"bar": bar, "note": pitch}
                if artic:
                    ev["artic"] = artic
                events.append(ev)
                continue
            i += 1
        return events


__all__ = ["SaxTokenizer"]
