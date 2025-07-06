from __future__ import annotations

from typing import List, Optional

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - optional
    torch = None  # type: ignore
    nn = object  # type: ignore

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception:  # pragma: no cover - optional
    AutoModelForCausalLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore


class PianoTransformer(nn.Module if torch is not None else object):
    """Tiny transformer for piano voicing generation."""

    def __init__(self, model_name: str) -> None:
        if AutoModelForCausalLM is None or AutoTokenizer is None or torch is None:
            raise RuntimeError("Install torch and transformers to use piano_ml")
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def encode(self, chord_label: str) -> List[int]:
        return [self.tokenizer.convert_tokens_to_ids(chord_label)]

    def sample_voicing(
        self, chord_label: str, prev_voicings: Optional[List[List[int]]] = None
    ) -> List[int]:
        if AutoModelForCausalLM is None or torch is None:
            raise RuntimeError("Install torch and transformers to use piano_ml")
        ids = torch.tensor([self.encode(chord_label)], dtype=torch.long)
        with torch.no_grad():
            generated = self.model.generate(ids, max_new_tokens=4)
        tokens = generated[0].tolist()[len(ids[0]) :]
        return tokens


__all__ = ["PianoTransformer"]
