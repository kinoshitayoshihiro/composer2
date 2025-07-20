from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
import yaml  # type: ignore
from torch.utils.data import DataLoader, Dataset

from utilities.duration_bucket import to_bucket


class ArticulationDataset(Dataset):
    def __init__(self, df: pd.DataFrame, label_map: dict[str, int]) -> None:
        self.groups = [g.sort_values("onset") for _, g in df.groupby("track_id")]
        self.label_map = label_map

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        g = self.groups[idx]
        pitch = torch.tensor(g["pitch"].to_numpy(), dtype=torch.long)
        dur = torch.tensor([to_bucket(d) for d in g["duration"]], dtype=torch.long)
        vel = torch.tensor(g["velocity"].to_numpy(), dtype=torch.float32)
        pedal = torch.tensor(g["pedal_state"].to_numpy(), dtype=torch.long)
        labels = torch.tensor(
            [self.label_map.get(str(a), 0) for a in g["articulation_label"]],
            dtype=torch.long,
        )
        return pitch, dur, vel, pedal, labels


def pad_collate(
    batch: list[tuple[torch.Tensor, ...]],
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]:
    pitches, durs, vels, pedals, labels = zip(*batch)
    max_len = max(x.size(0) for x in pitches)

    def _pad(
        seq: Iterable[torch.Tensor], fill: int = 0, dtype: torch.dtype = torch.long
    ) -> torch.Tensor:
        out = torch.full((len(batch), max_len), fill, dtype=dtype)
        for i, s in enumerate(seq):
            out[i, : s.size(0)] = s
        return out

    pitch = _pad(pitches)
    dur = _pad(durs)
    vel = _pad(vels, dtype=torch.float32)
    pedal = _pad(pedals)
    lbl = _pad(labels)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for i, s in enumerate(pitches):
        mask[i, : s.size(0)] = 1
    return (pitch, dur, vel, pedal), lbl, mask


class ArticulationDataModule(pl.LightningDataModule):
    def __init__(
        self,
        path: Path,
        schema: Path,
        batch_size: int = 8,
        train_pct: float = 0.8,
        num_workers: int = 0,
    ) -> None:
        super().__init__()
        self.path = Path(path)
        self.schema_path = Path(schema)
        self.batch_size = batch_size
        self.train_pct = train_pct
        self.num_workers = num_workers
        self.label_map: dict[str, int] = {}

    def setup(self, stage: str | None = None) -> None:
        df = pd.read_csv(self.path)
        with self.schema_path.open() as f:
            self.label_map = yaml.safe_load(f)
        n = int(len(df["track_id"].unique()) * self.train_pct)
        groups = df.groupby("track_id")
        train_ids = list(groups.groups.keys())[:n]
        train_df = df[df["track_id"].isin(train_ids)]
        val_df = df[~df["track_id"].isin(train_ids)]
        self.train_ds = ArticulationDataset(train_df, self.label_map)
        self.val_ds = ArticulationDataset(val_df, self.label_map)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=pad_collate,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=pad_collate,
        )
