from __future__ import annotations

import json
from typing import List, Dict

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
except Exception:  # pragma: no cover - optional dependency
    AutoModelForCausalLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    pipeline = None  # type: ignore


class TransformerBassGenerator:
    """Generate bass events using a Transformer model."""

    def __init__(self, model_name: str = "gpt2-music") -> None:
        if AutoModelForCausalLM is None:
            raise RuntimeError("transformers package required")
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)

    def _parse_events(self, text: str) -> List[Dict]:
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return [dict(ev) for ev in data]
        except Exception:
            pass
        return []

    def generate(self, prompt_events: List[Dict], bars: int) -> List[Dict]:
        prompt = json.dumps({"events": prompt_events, "bars": bars})
        out = self.pipe(prompt, max_new_tokens=64, num_return_sequences=1)[0]["generated_text"]
        generated = out[len(prompt) :].strip()
        return self._parse_events(generated)

__all__ = ["TransformerBassGenerator"]
