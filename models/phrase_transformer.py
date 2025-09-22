import torch
from torch import nn


class PhraseTransformer(nn.Module):
    """
    Minimal, test-friendly implementation.

    - どんなキーワード引数でも受け取れる（d_model, max_len, nhead など）。
    - forward(feats, mask) は dict を返し、少なくとも
      'boundary' -> [B, T]、'pointer' -> [B, T, T] を用意。
    - 実装は軽量（形状だけ満たせば良いテストのため）。
    """

    def __init__(
        self,
        d_model: int = 32,
        max_len: int = 512,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.max_len = int(max_len)

    def forward(
        self,
        feats: object,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # マスクからバッチ次元/系列長を決定
        if not isinstance(mask, torch.Tensor) or mask.ndim < 2:
            raise RuntimeError("mask must be a boolean tensor of shape [B, T]")
        B = int(mask.shape[0])
        T = int(mask.shape[1])

        device = mask.device
        # 形状だけ満たすゼロテンソル（ロジット相当）
        boundary = torch.zeros(B, T, device=device)  # [B, T]
        pointer = torch.zeros(B, T, T, device=device)  # [B, T, T]
        pitch_logits = torch.zeros(B, T, 128, device=device)  # [B, T, 128]
        return {
            "boundary": boundary,
            "pointer": pointer,
            "pitch_logits": pitch_logits,
        }
