from __future__ import annotations

import torch
from torch import nn
import math
import pytorch_lightning as pl

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return x

class DurationTransformer(pl.LightningModule):
    def __init__(self, d_model: int = 64, max_len: int = 16) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.d_model = d_model
        self.register_buffer("max_len", torch.tensor(max_len))
        self.pitch_emb = nn.Embedding(12, d_model // 4)
        self.pos_emb = nn.Embedding(max_len, d_model // 4)
        self.dur_proj = nn.Linear(1, d_model // 4)
        self.vel_proj = nn.Linear(1, d_model // 4)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, batch_first=True, dropout=0.0
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=4)
        self.pos_enc = PositionalEncoding(d_model, max_len + 1)
        self.fc = nn.Linear(d_model, 1)
        self.criterion = nn.SmoothL1Loss()

    def forward(self, feats: dict[str, torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
        pos_len = int(self.max_len.item())
        pos_ids = feats['position_in_bar'].clamp(max=pos_len - 1)
        feats['position_in_bar'] = pos_ids
        dur = self.dur_proj(feats['duration'].unsqueeze(-1))
        vel = self.vel_proj(feats['velocity'].unsqueeze(-1))
        pc = self.pitch_emb(feats['pitch_class'])
        pos = self.pos_emb(feats['position_in_bar'])
        x = torch.cat([dur, vel, pc, pos], dim=-1)
        cls = self.cls.expand(x.size(0), 1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x * math.sqrt(self.d_model)
        x = self.pos_enc(x)
        pad_mask = torch.cat([
            torch.ones(mask.size(0), 1, dtype=torch.bool, device=mask.device),
            mask,
        ], dim=1)
        src_key_padding_mask = ~pad_mask  # True = pad
        h = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        out = self.fc(h[:, 1:]).squeeze(-1)
        return out

    def validation_step(self, batch, batch_idx):
        feats, targets, mask = batch
        pred = self(feats, mask)
        loss = self.criterion(pred[mask], targets[mask])
        self.log('val_loss', loss, prog_bar=True)

    def training_step(self, batch, batch_idx):
        feats, targets, mask = batch
        pred = self(feats, mask)
        loss = self.criterion(pred[mask], targets[mask])
        self.log('loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


def predict(feats: dict[str, torch.Tensor], mask: torch.Tensor, model: DurationTransformer) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        out = model(feats, mask)
    return out

__all__ = ['DurationTransformer', 'PositionalEncoding', 'predict']
