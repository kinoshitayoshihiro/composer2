from __future__ import annotations

import pytorch_lightning as pl
import torch
from torch import nn
from TorchCRF import CRF


class ArticulationTagger(pl.LightningModule):
    """BiGRU-CRF based articulation tagger."""

    def __init__(self, num_labels: int) -> None:
        super().__init__()
        emb_pitch = 16
        emb_dur = 4
        emb_pedal = 4
        emb_vel = 4
        self.pitch_emb = nn.Embedding(128, emb_pitch)
        self.dur_emb = nn.Embedding(16, emb_dur)
        self.pedal_emb = nn.Embedding(3, emb_pedal)
        self.vel_proj = nn.Linear(1, emb_vel)
        input_size = emb_pitch + emb_dur + emb_pedal + emb_vel
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=128,
            num_layers=2,
            dropout=0.2,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(128 * 2, num_labels)
        self.crf = CRF(num_labels)
        if not hasattr(self.crf, "decode"):
            self.crf.decode = self.crf.viterbi_decode  # type: ignore[attr-defined]

    def forward(
        self,
        pitch: torch.Tensor,
        dur: torch.Tensor,
        vel: torch.Tensor,
        pedal: torch.Tensor,
    ) -> torch.Tensor:
        p = self.pitch_emb(pitch)
        d = self.dur_emb(dur)
        ped = self.pedal_emb(pedal)
        v = self.vel_proj(vel.unsqueeze(-1).float())
        x = torch.cat([p, d, v, ped], dim=-1)
        out, _ = self.gru(x)
        return self.fc(out)

    def training_step(
        self,
        batch: tuple[tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        feats, labels, mask = batch
        emissions = self(*feats)
        loss = -self.crf(emissions, labels, mask)
        self.log("loss", loss)
        return loss

    def decode_batch(self, batch: dict[str, torch.Tensor]) -> list[list[int]]:
        """Decode batch using CRF Viterbi algorithm."""
        with torch.no_grad():
            pitch = batch["pitch"]
            dur = batch["qlen"]
            vel = batch["velocity"]
            pedal = batch["pedal"]
            mask = batch.get("mask") or batch.get("pad_mask")
            if mask is None:
                mask = torch.ones_like(pitch, dtype=torch.bool)

            emissions = self(pitch, dur, vel, pedal)
            return self.crf.decode(emissions, mask)

    def configure_optimizers(self) -> dict:
        opt = torch.optim.Adam(self.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=2)
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "monitor": "val_loss"},
        }
