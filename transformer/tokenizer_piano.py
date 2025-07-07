from __future__ import annotations

import json
import warnings


class PianoTokenizer:
    """Minimal tokenizer for piano events."""

    def __init__(self) -> None:
        specials = [
            "<BAR>",
            "<TS_4_4>",
            "<LH>",
            "<RH>",
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
        export_vocab(path, self.vocab)

    def encode(self, events: list[dict[str, object]]) -> list[int]:
        tokens: list[int] = [self.token_to_id["<TS_4_4>"]]
        current_bar = -1
        unk = 0
        for ev in events:
            bar = int(ev.get("bar", 0))
            while bar > current_bar:
                tokens.append(self.token_to_id["<BAR>"])
                current_bar += 1
            hand = ev.get("hand", "rh")
            tokens.append(self.token_to_id["<LH>" if str(hand).lower() == "lh" else "<RH>"])
            pitch_token = f"P{int(ev.get('note', 60))}"
            if pitch_token in self.token_to_id:
                tokens.append(self.token_to_id[pitch_token])
            else:
                tokens.append(self.token_to_id["<UNK>"])
                unk += 1
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
            if tok in ("<LH>", "<RH>"):
                hand = "lh" if tok == "<LH>" else "rh"
                i += 1
                if i >= len(ids):
                    break
                pitch_tok = self.id_to_token[ids[i]]
                if pitch_tok == "<UNK>":
                    i += 1
                    continue
                if not pitch_tok.startswith("P"):
                    i += 1
                    continue
                pitch = int(pitch_tok[1:])
                i += 1
                if i < len(ids) and self.id_to_token[ids[i]] == "<VELO_80>":
                    i += 1
                events.append({"bar": bar, "note": pitch, "hand": hand})
                continue
            i += 1
        return events


def export_vocab(path: str, vocab: list[str] | None = None) -> None:
    """Export ``vocab`` or :class:`PianoTokenizer` vocab to ``path``."""
    if vocab is None:
        vocab = PianoTokenizer().vocab
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(vocab, fh)


__all__ = ["PianoTokenizer", "export_vocab"]
