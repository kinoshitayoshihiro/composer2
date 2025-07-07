from __future__ import annotations

from typing import Optional, Tuple

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - optional
    torch = None  # type: ignore
    nn = object  # type: ignore

try:
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import GPT2Config, GPT2LMHeadModel
except Exception:  # pragma: no cover - optional
    GPT2Config = None  # type: ignore
    GPT2LMHeadModel = None  # type: ignore
    LoraConfig = None  # type: ignore
    TaskType = None  # type: ignore
    get_peft_model = None  # type: ignore


class PianoTransformer(nn.Module if torch is not None else object):
    """GPT-2 based transformer with LoRA for piano events."""

    def __init__(self, vocab_size: int, rank: int = 4) -> None:
        if GPT2Config is None or GPT2LMHeadModel is None or get_peft_model is None or torch is None:
            raise RuntimeError("Install torch, transformers and peft to use PianoTransformer")
        super().__init__()
        config = GPT2Config(
            vocab_size=vocab_size,
            n_layer=8,
            n_head=8,
            n_embd=512,
        )
        base = GPT2LMHeadModel(config)
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=rank,
            lora_alpha=16,
            target_modules=["c_attn"],
            inference_mode=False,
        )
        self.model = get_peft_model(base, lora_cfg)

    def forward(
        self, input_ids: torch.Tensor, past_key_values: Optional[Tuple[Tuple[torch.Tensor, ...], ...]] = None
    ) -> torch.Tensor:
        if torch is None:
            raise RuntimeError("torch not available")
        outputs = self.model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)
        return outputs.logits


__all__ = ["PianoTransformer"]
