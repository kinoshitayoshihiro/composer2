from __future__ import annotations

import math
from typing import Sequence, Dict, Any

try:
    import torch
    from torch import nn
    import pytorch_lightning as pl
except Exception:  # pragma: no cover - optional dep
    torch = None  # type: ignore
    nn = object  # type: ignore
    pl = object  # type: ignore


class GrooveTransformer(pl.LightningModule if torch is not None else object):
    """Minimal multi-part Transformer model."""

    def __init__(
        self,
        vocab_sizes: Dict[str, int],
        *,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        lr: float = 1e-3,
        resolution: int = 16,
    ) -> None:
        if torch is None:
            raise RuntimeError("torch not available")
        super().__init__()
        self.save_hyperparameters()
        self.parts = list(vocab_sizes.keys())
        self.embeddings = nn.ModuleDict({p: nn.Embedding(v, d_model) for p, v in vocab_sizes.items()})
        self.pos_emb = nn.Embedding(1024, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.heads = nn.ModuleDict({p: nn.Linear(d_model, vocab_sizes[p]) for p in self.parts})
        self.loss_fn = nn.CrossEntropyLoss()

    def configure_optimizers(self) -> Any:
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def forward(self, tokens: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        seq_len = tokens[self.parts[0]].size(1)
        pos = torch.arange(seq_len, device=tokens[self.parts[0]].device)
        pos_emb = self.pos_emb(pos)[None, :, :]
        emb = 0
        for p in self.parts:
            emb = emb + self.embeddings[p](tokens[p])
        hidden = self.encoder(emb + pos_emb)
        return {p: self.heads[p](hidden) for p in self.parts}

    def training_step(self, batch: Dict[str, Dict[str, torch.Tensor]], batch_idx: int) -> torch.Tensor:  # type: ignore[override]
        inputs = batch["input"]
        targets = batch["target"]
        outputs = self(inputs)
        loss = 0.0
        for p in self.parts:
            logits = outputs[p][:, :-1, :]
            tgt = targets[p][:, 1:]
            loss_p = self.loss_fn(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
            self.log(f"loss_{p}", loss_p, prog_bar=False)
            preds = logits.argmax(-1)
            acc = (preds == tgt).float().mean()
            self.log(f"acc_{p}", acc, prog_bar=False)
            loss = loss + loss_p
        return loss


class MultiPartDataset(torch.utils.data.Dataset if torch is not None else object):
    """Simple dataset of aligned multi-part token sequences."""

    def __init__(self, sequences: Sequence[Dict[str, Sequence[int]]], parts: Sequence[str]) -> None:
        if torch is None:
            raise RuntimeError("torch not available")
        self.sequences = list(sequences)
        self.parts = list(parts)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:  # type: ignore[override]
        item = self.sequences[idx]
        return {p: torch.tensor(item[p], dtype=torch.long) for p in self.parts}


def collate_multi_part(batch: Sequence[Dict[str, torch.Tensor]], parts: Sequence[str], pad: int = 0) -> Dict[str, torch.Tensor]:
    if torch is None:
        raise RuntimeError("torch not available")
    max_len = max(x[p].size(0) for x in batch for p in parts)
    out = {p: [] for p in parts}
    for sample in batch:
        for p in parts:
            seq = sample[p]
            if seq.size(0) < max_len:
                pad_len = max_len - seq.size(0)
                seq = torch.cat([seq, torch.full((pad_len,), pad, dtype=torch.long)], dim=0)
            out[p].append(seq)
    return {p: torch.stack(out[p]) for p in parts}


__all__ = ["GrooveTransformer", "MultiPartDataset", "collate_multi_part"]
