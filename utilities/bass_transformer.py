from __future__ import annotations

from typing import Any, List, Optional

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore
    nn = object  # type: ignore

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraModel
except Exception:  # pragma: no cover - optional dependency
    AutoModelForCausalLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    LoraModel = None  # type: ignore


class BassTransformer(nn.Module if torch is not None else object):
    """Simple transformer-based sampler for bass generation."""

    def __init__(self, model_name: str, lora_path: Optional[str] = None) -> None:
        if AutoModelForCausalLM is None or AutoTokenizer is None or torch is None:
            raise RuntimeError("Install torch or use --backend ngram")
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        special = ["<straight8>", "<swing16>"]
        added = self.tokenizer.add_special_tokens({"additional_special_tokens": special})
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        if added:
            self.model.resize_token_embeddings(len(self.tokenizer))
        if lora_path and LoraModel is not None:
            self.model = LoraModel.from_pretrained(self.model, lora_path)

    def encode(self, events: List[dict]) -> List[int]:
        """Encode note events to token ids."""
        return [int(ev.get("pitch", 0)) for ev in events]

    def sample(
        self,
        sequence: List[int],
        top_k: int,
        temperature: float,
        rhythm_schema: Optional[str] = None,
    ) -> List[int]:
        """Sample next 16 tokens conditioned on ``sequence``."""
        if AutoModelForCausalLM is None or torch is None:
            raise RuntimeError("Install torch or use --backend ngram")
        input_ids = torch.tensor([sequence], dtype=torch.long)
        if rhythm_schema:
            tok_id = self.tokenizer.convert_tokens_to_ids(rhythm_schema)
            input_ids = torch.cat([torch.tensor([[tok_id]]), input_ids], dim=1)
        with torch.no_grad():
            generated = self.model.generate(
                input_ids,
                do_sample=True,
                top_k=top_k,
                temperature=temperature,
                max_new_tokens=16,
            )
        return generated[0].tolist()[input_ids.size(1):]


__all__ = ["BassTransformer"]
