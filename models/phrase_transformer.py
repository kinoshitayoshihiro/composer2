from __future__ import annotations
from typing import Any, Dict
import logging

try:
    import torch
    from torch import nn
except Exception:
    torch = None  # type: ignore
    nn = object   # type: ignore


class PhraseTransformer(nn.Module if torch is not None else object):
    """Lightweight fallback used by tests.

    - どんなハイパラでも受け取れる（d_model/max_len など）
    - forward(feats, mask) を呼ばれたときは (B, T) 形状の 'boundary' ロジットを返す
    """
    def __init__(
        self,
        d_model: int = 16,
        max_len: int = 128,
        nhead: int = 2,
        num_layers: int = 2,
        pitch_vocab_size: int = 128,
        vel_bins: int = 0,
        dur_bins: int = 0,
        duv_mode: str = "reg",
        **kwargs: Any,
    ) -> None:
        if torch is None:  # torch 未導入環境でも __init__ は通る
            self.d_model = int(d_model)
            self.max_len = int(max_len)
            return
        super().__init__()
        self.d_model = int(d_model)
        self.max_len = int(max_len)

    # PyTorch 互換の forward
    def forward(self, feats: Dict[str, "torch.Tensor"], mask: "torch.Tensor"):  # type: ignore[name-defined]
        if torch is None:
            # 極端な互換: mask を Python 配列とみなして空ロジットを返す（通常ここには来ない）
            try:
                t = len(mask[0])  # type: ignore[index]
                b = len(mask)
            except Exception:
                b, t = 1, 0
            return {"boundary": [[0.0] * t for _ in range(b)]}

        # mask は (B, T) を想定。1D の場合は (1, T) に昇格
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)
        b, t = mask.shape

        # 単純なダミーロジット（形状だけ保証）
        logits = torch.linspace(0, 1, steps=t, device=mask.device).repeat(b, 1)
        logits = logits.masked_fill(~mask.bool(), float("-inf"))
        return {"boundary": logits}
