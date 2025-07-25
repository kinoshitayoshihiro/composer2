from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


class ArticulationDataset(Dataset[dict[str, float | int]]):
    """Simple dataset wrapping a CSV exported by :func:`extract_from_midi`."""

    def __init__(self, csv_path: Path) -> None:
        df = pd.read_csv(csv_path)
        df["velocity"] = df["velocity"].clip(0.0, 1.0)
        self.df = df

        labels = torch.tensor(
            df["articulation_label"].fillna(0).astype(int).to_numpy(), dtype=torch.long
        )
        counts = torch.bincount(labels, minlength=int(labels.max()) + 1)
        weights = counts.sum() / counts.clamp(min=1)
        self.class_weight = weights.float()

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, float | int]:  # pragma: no cover
        row = self.df.iloc[idx]
        return {
            "pitch": int(row.pitch),
            "bucket": int(row.bucket),
            "pedal_state": int(row.pedal_state),
            "velocity": float(row.velocity),
            "qlen": float(row.duration),
            "label": int(row.articulation_label),
        }


def collate(batch: list[dict[str, float | int]]) -> dict[str, torch.Tensor]:
    keys = ["pitch", "bucket", "pedal_state", "velocity", "qlen", "label"]
    out = {k: torch.tensor([item[k] for item in batch]) for k in keys}
    out["pad_mask"] = torch.ones(len(batch), dtype=torch.bool)
    return out


class SeqArticulationDataset(Dataset[list[dict[str, float | int]]]):
    """Dataset yielding sorted note sequences per track."""

    def __init__(self, csv_path: Path) -> None:
        df = pd.read_csv(csv_path)
        df["velocity"] = df["velocity"].clip(0.0, 1.0)
        self.tracks = [g.sort_values("onset") for _, g in df.groupby("track_id")]

        labels = torch.tensor(
            df["articulation_label"].fillna(0).astype(int).to_numpy(),
            dtype=torch.long,
        )
        counts = torch.bincount(labels, minlength=int(labels.max()) + 1)
        weights = counts.sum() / counts.clamp(min=1)
        self.class_weight = weights.float()

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.tracks)

    def __getitem__(self, idx: int) -> list[dict[str, float | int]]:  # pragma: no cover
        df = self.tracks[idx]
        return [
            {
                "pitch": int(r.pitch),
                "bucket": int(r.bucket),
                "pedal_state": int(r.pedal_state),
                "velocity": float(r.velocity),
                "qlen": float(r.duration),
                "label": int(r.articulation_label),
            }
            for r in df.itertuples()
        ]


def seq_collate(batch: list[list[dict[str, float | int]]]) -> dict[str, torch.Tensor]:
    """Pad variable-length sequences and stack note features separately."""

    max_len = max(len(seq) for seq in batch)

    for seq in batch:
        for item in seq:
            item["pedal"] = item.get("pedal_state", 0)
            item["velocity"] = item.get("velocity", 0.0)
            item["qlen"] = item.get("qlen", 0)

    def pad(
        seq: list[dict[str, float | int]], key: str, fill: float | int
    ) -> list[float | int]:
        return [item[key] for item in seq] + [fill] * (max_len - len(seq))

    out = {
        "pitch": torch.tensor(
            [pad(seq, "pitch", 0) for seq in batch], dtype=torch.long
        ),
        "bucket": torch.tensor(
            [pad(seq, "bucket", 0) for seq in batch], dtype=torch.long
        ),
        "pedal": torch.tensor(
            [pad(seq, "pedal", 0) for seq in batch], dtype=torch.long
        ),
        "velocity": torch.tensor(
            [pad(seq, "vel", 0.0) for seq in batch], dtype=torch.float32
        ),
        "qlen": torch.tensor([pad(seq, "dur", 0) for seq in batch], dtype=torch.long),
        "labels": torch.tensor(
            [pad(seq, "label", 0) for seq in batch], dtype=torch.long
        ),
    }
    out["pad_mask"] = torch.tensor(
        [[1] * len(seq) + [0] * (max_len - len(seq)) for seq in batch], dtype=torch.bool
    )
    return out


def get_dataloader(
    csv_path: Path, batch_size: int, weighted: bool = True
) -> DataLoader[Any]:
    ds = SeqArticulationDataset(csv_path)
    sampler = None
    if weighted:
        seq_weights = []
        for df in ds.tracks:
            labels = df["articulation_label"].fillna(0).astype(int).to_numpy()
            w = ds.class_weight[labels].mean().item()
            seq_weights.append(w)
        sampler = WeightedRandomSampler(seq_weights, len(seq_weights), replacement=True)
    return DataLoader(
        ds, batch_size=batch_size, sampler=sampler, collate_fn=seq_collate
    )


__all__ = [
    "ArticulationDataset",
    "collate",
    "SeqArticulationDataset",
    "seq_collate",
    "get_dataloader",
]
