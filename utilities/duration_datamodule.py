from __future__ import annotations

from pathlib import Path
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader


class DurationDataset(Dataset):
    def __init__(self, df: pd.DataFrame, max_len: int = 16) -> None:
        self.groups = [g.sort_values("position") for _, g in df.groupby("bar")]
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, idx: int):
        g = self.groups[idx]
        L = len(g)
        pad = self.max_len - L
        # 'pitch'列を使い、その場で 'pitch_class' を計算する
        pc = torch.tensor(pitch_class_list + [0] * pad, dtype=torch.long)
        dur = torch.tensor(list(g["duration"]) + [0.0] * pad, dtype=torch.float32)
        vel = torch.tensor(list(g["velocity"]) + [0.0] * pad, dtype=torch.float32)
        pitch_class_list = [p % 12 for p in g["pitch"]]
        pos = torch.tensor(list(g["position"]) + [0] * pad, dtype=torch.long)
        mask = torch.zeros(self.max_len, dtype=torch.bool)
        mask[:L] = 1
        return (
            {
                "duration": dur,
                "velocity": vel,
                "pitch_class": pc,
                "position_in_bar": pos,
            },
            dur,
            mask,
        )


def collate_fn(batch):
    feats, targets, masks = zip(*batch)
    out_feats = {k: torch.stack([f[k] for f in feats]) for k in feats[0]}
    out_targets = torch.stack(targets)
    out_masks = torch.stack(masks)
    return out_feats, out_targets, out_masks


class DurationDataModule(pl.LightningDataModule):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.csv_path = Path(cfg.data.csv)
        self.batch_size = cfg.batch_size
        self.max_len = cfg.max_len

    def setup(self, stage: str | None = None) -> None:
        df = pd.read_csv(self.csv_path)
        df_train = df.sample(frac=0.9, random_state=42)
        df_val = df.drop(df_train.index)
        self.train_ds = DurationDataset(df_train, self.max_len)
        self.val_ds = DurationDataset(df_val, self.max_len)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )


__all__ = ["DurationDataModule", "collate_fn"]
