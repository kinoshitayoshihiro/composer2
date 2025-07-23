from __future__ import annotations

import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):  # type: ignore[misc]
    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class PhraseTransformer(nn.Module):  # type: ignore[misc]
    """Simple transformer encoder for phrase segmentation."""

    def __init__(self, d_model: int = 512, max_len: int = 128) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.pitch_emb = nn.Embedding(12, d_model // 4)
        self.pos_emb = nn.Embedding(max_len, d_model // 4)
        self.dur_proj = nn.Linear(1, d_model // 4)
        self.vel_proj = nn.Linear(1, d_model // 4)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=8, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=4)
        self.pos_enc = PositionalEncoding(d_model, max_len + 1)
        self.fc = nn.Linear(d_model, 1)

    def forward(
        self, feats: dict[str, torch.Tensor], mask: torch.Tensor
    ) -> torch.Tensor:
        pos_ids = feats["position"].clamp(max=self.max_len - 1)
        dur = self.dur_proj(feats["duration"].unsqueeze(-1))
        vel = self.vel_proj(feats["velocity"].unsqueeze(-1))
        pc = self.pitch_emb(feats["pitch_class"] % 12)
        pos = self.pos_emb(pos_ids)
        x = torch.cat([dur, vel, pc, pos], dim=-1)
        cls = self.cls.expand(x.size(0), 1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x * math.sqrt(self.d_model)
        x = self.pos_enc(x)
        pad_mask = torch.cat(
            [torch.ones(mask.size(0), 1, dtype=torch.bool, device=mask.device), mask],
            dim=1,
        )
        h = self.encoder(x, src_key_padding_mask=~pad_mask)
        out = self.fc(h[:, 1:]).squeeze(-1)
        return out


__all__ = ["PhraseTransformer"]
