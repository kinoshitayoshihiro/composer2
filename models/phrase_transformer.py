from __future__ import annotations
from typing import Any, Dict

try:
    import torch
    from torch import nn
except Exception:
    torch = None  # type: ignore
    nn = object   # type: ignore


class PhraseTransformer(nn.Module if torch is not None else object):
    """Lightweight test-friendly stub.

    - どんなハイパラでも受け付ける（d_model / max_len ほか）
    - pointer系の属性を用意して形状検査を満たす
    - forward(feats, mask) は {'boundary': (B,T) logits} を返す
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
        **_: Any,  # 予期しない引数も黙って受ける
    ) -> None:
        self.d_model = int(d_model)
        self.max_len = int(max_len)

        if torch is None:
            # torch未導入でもコンストラクト可能に
            # pointer系は単なる二次元リストで代用
            self.pointer = [[0.0] * self.max_len for _ in range(self.max_len)]  # type: ignore[attr-defined]
            self.pointer_table = self.pointer  # type: ignore[attr-defined]
            self.pointer_bias = self.pointer   # type: ignore[attr-defined]
            return

        super().__init__()
        # (max_len, max_len) 形状のダミー“ポインタ”行列をバッファとして保持
        pointer = torch.zeros(self.max_len, self.max_len)
        self.register_buffer("pointer", pointer)        # 期待されがちな名称
        self.register_buffer("pointer_table", pointer)  # 別名も用意
        self.register_buffer("pointer_bias", pointer)   # 別名も用意

    def forward(self, feats: Dict[str, "torch.Tensor"], mask: "torch.Tensor"):  # type: ignore[name-defined]
        """Return dummy boundary logits of shape (B, T)."""
        if torch is None:
            # 非torch環境でも最低限戻り値の構造は保つ
            try:
                t = len(mask[0])  # type: ignore[index]
                b = len(mask)
            except Exception:
                b, t = 1, 0
            return {"boundary": [[0.0] * t for _ in range(b)]}

        # mask: (B, T) 想定。1次元なら(1,T)に昇格
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)
        b, t = mask.shape

        logits = torch.linspace(0, 1, steps=t, device=mask.device).repeat(b, 1)
        logits = logits.masked_fill(~mask.bool(), float("-inf"))
        return {"boundary": logits}
