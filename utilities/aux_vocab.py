from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AuxVocab:
    """Simple vocabulary for auxiliary conditioning.

    Each unique set of key/value pairs is mapped to an integer id. The empty
    dictionary maps to id 0 (ANY) and unknown placeholders map to id 1.
    """

    str_to_id: dict[str, int] = field(default_factory=lambda: {"": 0, "<UNK>": 1})
    id_to_str: list[str] = field(default_factory=lambda: ["", "<UNK>"])

    def encode(self, cond: dict[str, str] | None) -> int:
        if not cond:
            return 0
        key = "|".join(f"{k}={v}" for k, v in sorted(cond.items()))
        return self.add(key)

    def add(self, key: str) -> int:
        idx = self.str_to_id.get(key)
        if idx is None:
            idx = len(self.id_to_str)
            self.str_to_id[key] = idx
            self.id_to_str.append(key)
        return idx

    def decode(self, idx: int) -> dict[str, str] | None:
        s = self.id_to_str[idx]
        if s == "":
            return {}
        if s == "<UNK>":
            return None
        return dict(part.split("=", 1) for part in s.split("|"))

    def to_json(self, path: Path) -> None:
        path.write_text(json.dumps(self.id_to_str))

    @classmethod
    def from_json(cls, path: Path) -> "AuxVocab":
        rev = json.loads(path.read_text())
        mapping = {s: i for i, s in enumerate(rev)}
        return cls(mapping, rev)
