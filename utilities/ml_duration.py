from __future__ import annotations

import torch
from torch import nn
import math
import pytorch_lightning as pl


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return x


class DurationTransformer(pl.LightningModule):
    def __init__(self, d_model: int = 64, max_len: int = 16, ff_dim: int = None) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.d_model = d_model
        self.ff_dim = ff_dim or (d_model * 4)  # Default to 4x like standard transformer
        self.register_buffer("max_len", torch.tensor(max_len))
        self.pitch_emb = nn.Embedding(12, d_model // 4)
        self.pos_emb = nn.Embedding(max_len, d_model // 4)
        self.dur_proj = nn.Linear(1, d_model // 4)
        self.vel_proj = nn.Linear(1, d_model // 4)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=self.ff_dim, batch_first=True, dropout=0.0
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=4)
        self.pos_enc = PositionalEncoding(d_model, max_len + 1)
        self.fc = nn.Linear(d_model, 1)
        self.criterion = nn.SmoothL1Loss()

    def forward(self, feats: dict[str, torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
        pos_len = int(self.max_len.item())
        pos_ids = feats["position_in_bar"].clamp(max=pos_len - 1)
        feats["position_in_bar"] = pos_ids
        dur = self.dur_proj(feats["duration"].unsqueeze(-1))
        vel = self.vel_proj(feats["velocity"].unsqueeze(-1))
        pc = self.pitch_emb(feats["pitch_class"])
        pos = self.pos_emb(feats["position_in_bar"])
        x = torch.cat([dur, vel, pc, pos], dim=-1)
        cls = self.cls.expand(x.size(0), 1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x * math.sqrt(self.d_model)
        x = self.pos_enc(x)
        pad_mask = torch.cat(
            [
                torch.ones(mask.size(0), 1, dtype=torch.bool, device=mask.device),
                mask,
            ],
            dim=1,
        )
        src_key_padding_mask = ~pad_mask  # True = pad
        h = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        out = self.fc(h[:, 1:]).squeeze(-1)
        return out

    def validation_step(self, batch, batch_idx):
        feats, targets, mask = batch
        pred = self(feats, mask)
        loss = self.criterion(pred[mask], targets[mask])
        self.log("val_loss", loss, prog_bar=True)

    def training_step(self, batch, batch_idx):
        feats, targets, mask = batch
        pred = self(feats, mask)
        loss = self.criterion(pred[mask], targets[mask])
        self.log("loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    @classmethod
    def _infer_hparams_from_sd(cls, state_dict: dict) -> dict:
        """Infer hyperparameters from state_dict structure"""
        # Infer d_model from cls parameter shape
        if "cls" in state_dict:
            d_model = state_dict["cls"].shape[-1]
        else:
            d_model = 64  # fallback

        # Infer max_len from pos_emb or pos_enc
        if "pos_emb.weight" in state_dict:
            max_len = state_dict["pos_emb.weight"].shape[0]
        elif "pos_enc.pe" in state_dict:
            max_len = state_dict["pos_enc.pe"].shape[1] - 1  # minus CLS token
        else:
            max_len = 16  # fallback

        # Infer ff_dim from encoder layers
        ff_dim = None
        if "encoder.layers.0.linear1.weight" in state_dict:
            ff_dim = state_dict["encoder.layers.0.linear1.weight"].shape[0]

        return {"d_model": d_model, "max_len": max_len, "ff_dim": ff_dim}

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "DurationTransformer":
        """Load model with automatic hyperparameter inference"""
        ckpt = torch.load(path, map_location=device)

        # Handle different checkpoint formats
        if isinstance(ckpt, dict):
            if "model" in ckpt:
                # Direct model format
                return ckpt["model"].eval()
            elif "state_dict" in ckpt:
                state_dict = ckpt["state_dict"]
            else:
                state_dict = ckpt
        else:
            state_dict = ckpt

        # Infer hyperparameters from state_dict
        hparams = cls._infer_hparams_from_sd(state_dict)

        # Create model with inferred parameters
        model = cls(**hparams)

        # Load state dict with error handling
        try:
            model.load_state_dict(state_dict, strict=True)
        except Exception as e:
            print(f"WARNING: Could not load duration model state_dict: {e}")
            print(f"Model will use random weights")

        return model


def predict(
    feats: dict[str, torch.Tensor], mask: torch.Tensor, model: DurationTransformer
) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        out = model(feats, mask)
    return out


__all__ = ["DurationTransformer", "PositionalEncoding", "predict"]
