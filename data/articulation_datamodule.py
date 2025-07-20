from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from .articulation_dataset import SeqArticulationDataset, seq_collate


class ArticulationDataModule(pl.LightningDataModule):
    """DataModule wrapping :class:`SeqArticulationDataset`."""

    def __init__(
        self,
        csv_path: Path,
        batch_size: int = 16,
        weighted: bool = False,
        val_pct: float = 0.1,
        num_workers: int = 0,
    ) -> None:
        super().__init__()
        self.csv_path = Path(csv_path)
        self.batch_size = batch_size
        self.weighted = weighted
        self.val_pct = val_pct
        self.num_workers = num_workers
        self._class_weight: torch.Tensor | None = None

    @property
    def class_weight(self) -> torch.Tensor:
        assert self._class_weight is not None
        return self._class_weight

    def setup(self, stage: str | None = None) -> None:
        ds = SeqArticulationDataset(self.csv_path)
        self._class_weight = ds.class_weight
        n = len(ds)
        split = int(n * (1 - self.val_pct))
        indices = list(range(n))
        self.train_ds = cast(
            Subset[SeqArticulationDataset], Subset(ds, indices[:split])
        )
        self.val_ds = cast(Subset[SeqArticulationDataset], Subset(ds, indices[split:]))
        self.test_ds = self.val_ds

    def _sampler(
        self, subset: Subset[SeqArticulationDataset]
    ) -> WeightedRandomSampler | None:
        if not self.weighted:
            return None
        ds = cast(SeqArticulationDataset, subset.dataset)
        cache = self.csv_path.with_suffix(".weights.pt")
        if cache.exists() and cache.stat().st_mtime > self.csv_path.stat().st_mtime:
            all_weights = torch.load(cache)
        else:
            seq_weights = []
            for df in ds.tracks:
                labels = df["articulation_label"].fillna(0).astype(int).to_numpy()
                w = ds.class_weight[labels].mean().item()
                seq_weights.append(w)
            all_weights = torch.tensor(seq_weights, dtype=torch.float32)
            torch.save(all_weights, cache)
        weights = all_weights[torch.tensor(subset.indices)]
        return WeightedRandomSampler(weights.tolist(), len(weights), replacement=True)

    def train_dataloader(self) -> DataLoader[Any]:
        sampler = self._sampler(self.train_ds)
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            sampler=sampler,
            shuffle=sampler is None,
            collate_fn=seq_collate,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            collate_fn=seq_collate,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            collate_fn=seq_collate,
            num_workers=self.num_workers,
        )


__all__ = ["ArticulationDataModule"]
