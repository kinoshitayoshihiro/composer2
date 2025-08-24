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

    def __init__(
        self,
        d_model: int = 512,
        max_len: int = 128,
        *,
        section_vocab_size: int = 0,
        mood_vocab_size: int = 0,
        vel_bucket_size: int = 0,
        dur_bucket_size: int = 0,
        duv_mode: str = "reg",
        vel_bins: int = 0,
        dur_bins: int = 0,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.pitch_emb = nn.Embedding(12, d_model // 4)
        self.pos_emb = nn.Embedding(max_len, d_model // 4)
        self.dur_proj = nn.Linear(1, d_model // 4)
        self.vel_proj = nn.Linear(1, d_model // 4)
        extra_dim = 0
        if section_vocab_size:
            self.section_emb = nn.Embedding(section_vocab_size, 16)
            extra_dim += 16
        else:
            self.section_emb = None
        if mood_vocab_size:
            self.mood_emb = nn.Embedding(mood_vocab_size, 16)
            extra_dim += 16
        else:
            self.mood_emb = None
        if vel_bucket_size:
            self.vel_bucket_emb = nn.Embedding(vel_bucket_size, 8)
            extra_dim += 8
        else:
            self.vel_bucket_emb = None
        if dur_bucket_size:
            self.dur_bucket_emb = nn.Embedding(dur_bucket_size, 8)
            extra_dim += 8
        else:
            self.dur_bucket_emb = None
        self.feat_proj = nn.Linear(d_model + extra_dim, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, dropout=dropout
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.pos_enc = PositionalEncoding(d_model, max_len + 1)
        self.head_boundary = nn.Linear(d_model, 1)
        self.head_vel_reg = (
            nn.Linear(d_model, 1) if duv_mode in {"reg", "both"} else None
        )
        self.head_dur_reg = (
            nn.Linear(d_model, 1) if duv_mode in {"reg", "both"} else None
        )
        self.head_vel_cls = (
            nn.Linear(d_model, vel_bins) if duv_mode in {"cls", "both"} else None
        )
        self.head_dur_cls = (
            nn.Linear(d_model, dur_bins) if duv_mode in {"cls", "both"} else None
        )

    def forward(
        self, feats: dict[str, torch.Tensor], mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        pos_ids = feats["position"].clamp(max=self.max_len - 1)
        dur = self.dur_proj(feats["duration"].unsqueeze(-1))
        vel = self.vel_proj(feats["velocity"].unsqueeze(-1))
        pc = self.pitch_emb(feats["pitch_class"] % 12)
        pos = self.pos_emb(pos_ids)
        parts = [dur, vel, pc, pos]
        if self.section_emb is not None and "section" in feats:
            parts.append(self.section_emb(feats["section"]))
        if self.mood_emb is not None and "mood" in feats:
            parts.append(self.mood_emb(feats["mood"]))
        if self.vel_bucket_emb is not None and "vel_bucket" in feats:
            parts.append(self.vel_bucket_emb(feats["vel_bucket"]))
        if self.dur_bucket_emb is not None and "dur_bucket" in feats:
            parts.append(self.dur_bucket_emb(feats["dur_bucket"]))
        x = torch.cat(parts, dim=-1)
        x = self.feat_proj(x)
        cls = self.cls.expand(x.size(0), 1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x * math.sqrt(self.d_model)
        x = self.pos_enc(x)
        pad_mask = torch.cat(
            [torch.ones(mask.size(0), 1, dtype=torch.bool, device=mask.device), mask],
            dim=1,
        )
        h = self.encoder(x, src_key_padding_mask=~pad_mask)
        h = h[:, 1:]
        outputs: dict[str, torch.Tensor] = {}
        outputs["boundary"] = self.head_boundary(h).squeeze(-1)
        if self.head_vel_reg is not None:
            outputs["vel_reg"] = self.head_vel_reg(h).squeeze(-1)
        if self.head_dur_reg is not None:
            outputs["dur_reg"] = self.head_dur_reg(h).squeeze(-1)
        if self.head_vel_cls is not None:
            outputs["vel_cls"] = self.head_vel_cls(h)
        if self.head_dur_cls is not None:
            outputs["dur_cls"] = self.head_dur_cls(h)
        return outputs


__all__ = ["PhraseTransformer"]
