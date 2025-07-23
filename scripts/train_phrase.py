from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from models.phrase_transformer import PhraseTransformer


class PhraseLSTM(nn.Module):  # type: ignore[misc]
    def __init__(self, d_model: int = 128, max_len: int = 128) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.pitch_emb = nn.Embedding(12, d_model // 4)
        self.pos_emb = nn.Embedding(max_len, d_model // 4)
        self.dur_proj = nn.Linear(1, d_model // 4)
        self.vel_proj = nn.Linear(1, d_model // 4)
        self.lstm = nn.LSTM(
            d_model, d_model // 2, num_layers=2, batch_first=True, bidirectional=True
        )
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
        packed = nn.utils.rnn.pack_padded_sequence(
            x, mask.sum(dim=1).cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            out, batch_first=True, total_length=self.max_len
        )
        out = self.fc(out).squeeze(-1)
        return out


class PhraseDataset(Dataset):  # type: ignore[misc]
    def __init__(self, df: pd.DataFrame, max_len: int = 32) -> None:
        self.groups = [g.sort_values("pos") for _, g in df.groupby("bar")]
        self.max_len = max_len

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.groups)

    def __getitem__(
        self, idx: int
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        g = self.groups[idx]
        L = len(g)
        pad = self.max_len - L
        pc = torch.tensor(g["pitch"].tolist() + [0] * pad, dtype=torch.long)
        vel = torch.tensor(g["velocity"].tolist() + [0] * pad, dtype=torch.float32)
        dur = torch.tensor(g["duration"].tolist() + [0] * pad, dtype=torch.float32)
        pos = torch.tensor(g["pos"].tolist() + [0] * pad, dtype=torch.long)
        y = torch.tensor(g["boundary"].tolist() + [0] * pad, dtype=torch.float32)
        mask = torch.zeros(self.max_len, dtype=torch.bool)
        mask[:L] = 1
        return (
            {"pitch_class": pc, "velocity": vel, "duration": dur, "position": pos},
            y,
            mask,
        )


def collate_fn(
    batch: list[tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]],
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    feats, y, mask = zip(*batch)
    out_feats = {k: torch.stack([f[k] for f in feats]) for k in feats[0]}
    return out_feats, torch.stack(y), torch.stack(mask)


def train_model(
    train_csv: Path, val_csv: Path, epochs: int, arch: str, out: Path
) -> float:
    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)
    ds_train = PhraseDataset(df_train)
    ds_val = PhraseDataset(df_val)
    dl_train = DataLoader(ds_train, batch_size=8, shuffle=True, collate_fn=collate_fn)
    dl_val = DataLoader(ds_val, batch_size=8, shuffle=False, collate_fn=collate_fn)
    if arch == "lstm":
        model: nn.Module = PhraseLSTM()
    else:
        model = PhraseTransformer()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        model.train()
        for feats, y, mask in dl_train:
            opt.zero_grad()
            logits = model(feats, mask)
            loss = crit(logits[mask], y[mask])
            loss.backward()
            opt.step()
    model.eval()
    preds: list[float] = []
    trues: list[float] = []
    with torch.no_grad():
        for feats, y, mask in dl_val:
            logits = model(feats, mask)
            p = torch.sigmoid(logits[mask]) > 0.5
            preds.extend(p.cpu().tolist())
            trues.extend(y[mask].cpu().tolist())
    f1: float = float(f1_score(trues, preds))
    torch.save(model.state_dict(), out)
    metrics_path = out.parent / "metrics.json"
    metrics_path.write_text(json.dumps({"f1": f1}))
    print(json.dumps({"f1": f1}))
    return f1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_csv", type=Path)
    parser.add_argument("val_csv", type=Path)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--out", type=Path, default=Path("phrase.ckpt"))
    parser.add_argument(
        "--arch", choices=["transformer", "lstm"], default="transformer"
    )
    args = parser.parse_args(argv)
    f1 = train_model(args.train_csv, args.val_csv, args.epochs, args.arch, args.out)
    return 0 if f1 >= 0.91 else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
