from __future__ import annotations

from typing import List, Optional

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore
    nn = object  # type: ignore

try:
    from transformers import GPT2Config, GPT2LMHeadModel
    from peft import PeftModel
except Exception:  # pragma: no cover - optional dependency
    GPT2Config = None  # type: ignore
    GPT2LMHeadModel = None  # type: ignore
    PeftModel = None  # type: ignore

from transformer.tokenizer_piano import PianoTokenizer


class PianoMLGenerator(nn.Module if torch is not None else object):
    """Wrapper around :class:`PianoTransformer` for inference."""

    def __init__(self, model_path: str, *, temperature: float = 0.9) -> None:
        if torch is None or GPT2Config is None or GPT2LMHeadModel is None or PeftModel is None:
            raise RuntimeError("Install torch, transformers and peft to use PianoMLGenerator")
        super().__init__()
        self.tokenizer = PianoTokenizer()
        config = GPT2Config(
            vocab_size=len(self.tokenizer.vocab),
            n_layer=8,
            n_head=8,
            n_embd=512,
        )
        base = GPT2LMHeadModel(config)
        self.model = PeftModel.from_pretrained(base, model_path)
        self.model.eval()
        self.temperature = float(temperature)

    def _generate_tokens(self, tokens: List[int], max_new_tokens: int) -> List[int]:
        ids = torch.tensor([tokens], dtype=torch.long)
        with torch.no_grad():
            out = self.model.generate(
                ids,
                do_sample=True,
                temperature=self.temperature,
                max_new_tokens=max_new_tokens,
            )
        return out[0].tolist()[len(tokens):]

    def generate(
        self,
        prompt_events: Optional[List[dict[str, object]]] = None,
        *,
        max_bars: int = 8,
        temperature: Optional[float] = None,
    ) -> List[dict[str, object]]:
        """Generate up to ``max_bars`` of events from ``prompt_events``."""

        if temperature is not None:
            self.temperature = float(temperature)
        tokens = self.tokenizer.encode(prompt_events or [])
        new_tokens = self._generate_tokens(tokens, max_new_tokens=max_bars * 16)
        return self.tokenizer.decode(new_tokens)

    def step(self, context_events: List[dict[str, object]]) -> List[dict[str, object]]:
        """Generate the next bar of events after ``context_events``."""

        last_bar = context_events[-1]["bar"] if context_events else -1
        tokens = self.tokenizer.encode(context_events)
        new_tokens = self._generate_tokens(tokens, max_new_tokens=16)
        events = self.tokenizer.decode(new_tokens)
        return [ev for ev in events if int(ev.get("bar", last_bar + 1)) == last_bar + 1]


__all__ = ["PianoMLGenerator"]
