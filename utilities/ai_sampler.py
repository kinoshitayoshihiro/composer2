from __future__ import annotations

import json

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
except Exception:  # pragma: no cover - optional dependency
    AutoModelForCausalLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    pipeline = None  # type: ignore


class TransformerBassGenerator:
    """Generate bass events using a Transformer model."""

    def __init__(self, model_name: str = "gpt2-medium", rhythm_schema: str | None = None) -> None:
        if AutoModelForCausalLM is None:
            raise RuntimeError("transformers package required")
        self.model_name = model_name
        self.rhythm_schema = rhythm_schema or ""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)

    def _parse_events(self, text: str) -> list[dict]:
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return [dict(ev) for ev in data]
        except Exception:
            pass
        return []

    def generate(self, prompt_events: list[dict], bars: int) -> list[dict]:
        prompt = json.dumps({"events": prompt_events, "bars": bars})
        if self.rhythm_schema:
            prompt = f"{self.rhythm_schema} " + prompt
        max_tokens = min(1024, bars * 32)
        out = self.pipe(
            prompt,
            max_new_tokens=max_tokens,
            num_return_sequences=1,
        )[0]["generated_text"]
        generated = out[len(prompt) :].strip()
        return self._parse_events(generated)

__all__ = ["TransformerBassGenerator"]
