from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import logging
import random
import time
from contextlib import nullcontext

try:  # optional dependency
    import numpy as np
except Exception:  # pragma: no cover - numpy missing
    np = None
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
try:  # optional
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    SummaryWriter = None

from models.phrase_transformer import PhraseTransformer


def setup_env(seed: int = 42) -> tuple[torch.device, bool]:
    """Seed RNGs and select compute device.

    Returns the chosen ``torch.device`` and whether automatic mixed precision
    should be used.
    """

    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    amp = device.type == "cuda"
    return device, amp


class PhraseLSTM(nn.Module):  # type: ignore[misc]
    def __init__(
        self,
        d_model: int = 128,
        max_len: int = 128,
        *,
        section_vocab_size: int = 0,
        mood_vocab_size: int = 0,
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
        self.feat_proj = nn.Linear(d_model + extra_dim, d_model)
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
        parts = [dur, vel, pc, pos]
        if self.section_emb is not None and "section" in feats:
            parts.append(self.section_emb(feats["section"]))
        if self.mood_emb is not None and "mood" in feats:
            parts.append(self.mood_emb(feats["mood"]))
        x = torch.cat(parts, dim=-1)
        x = self.feat_proj(x)
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
    def __init__(
        self,
        rows: list[dict[str, str]],
        max_len: int = 32,
        section_vocab: dict[str, int] | None = None,
        mood_vocab: dict[str, int] | None = None,
    ) -> None:
        by_bar: dict[int, list[dict[str, str]]] = {}
        for r in rows:
            bar = int(r["bar"])
            by_bar.setdefault(bar, []).append(r)
        self.groups = [
            sorted(g, key=lambda r: int(r["pos"])) for bar, g in sorted(by_bar.items())
        ]
        self.max_len = max_len
        self.section_vocab = section_vocab
        self.mood_vocab = mood_vocab

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.groups)

    def __getitem__(
        self, idx: int
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        g = self.groups[idx]
        L = len(g)
        pad = self.max_len - L
        pc = torch.tensor([int(r["pitch"]) for r in g] + [0] * pad, dtype=torch.long)
        vel = torch.tensor([float(r["velocity"]) for r in g] + [0] * pad, dtype=torch.float32)
        dur = torch.tensor([float(r["duration"]) for r in g] + [0] * pad, dtype=torch.float32)
        pos = torch.tensor([int(r["pos"]) for r in g] + [0] * pad, dtype=torch.long)
        y = torch.tensor([float(r["boundary"]) for r in g] + [0] * pad, dtype=torch.float32)
        feats = {
            "pitch_class": pc,
            "velocity": vel,
            "duration": dur,
            "position": pos,
        }
        if self.section_vocab:
            sec = [self.section_vocab.get(r.get("section", ""), 0) for r in g] + [0] * pad
            feats["section"] = torch.tensor(sec, dtype=torch.long)
        if self.mood_vocab:
            md = [self.mood_vocab.get(r.get("mood", ""), 0) for r in g] + [0] * pad
            feats["mood"] = torch.tensor(md, dtype=torch.long)
        mask = torch.zeros(self.max_len, dtype=torch.bool)
        mask[:L] = 1
        return feats, y, mask


def collate_fn(
    batch: list[tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]],
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    feats, y, mask = zip(*batch)
    out_feats = {k: torch.stack([f[k] for f in feats]) for k in feats[0]}
    return out_feats, torch.stack(y), torch.stack(mask)


def load_csv_rows(path: Path, required: set[str]) -> list[dict[str, str]]:
    """Read *path* and ensure required columns exist."""
    with path.open() as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise SystemExit(f"CSV missing required columns {required}")
        return [row for row in reader]


def f1_score(trues: list[int], preds: list[int]) -> float:
    """Compute binary F1 score without external dependencies."""
    tp = sum(1 for t, p in zip(trues, preds) if t and p)
    fp = sum(1 for t, p in zip(trues, preds) if not t and p)
    fn = sum(1 for t, p in zip(trues, preds) if t and not p)
    denom = 2 * tp + fp + fn
    return 2 * tp / denom if denom else 0.0


def train_model(
    train_csv: Path,
    val_csv: Path,
    epochs: int,
    arch: str,
    out: Path,
    *,
    batch_size: int = 8,
    d_model: int = 512,
    max_len: int = 128,
    num_workers: int = 0,
    pin_memory: bool = False,
    grad_clip: float = 1.0,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    scheduler: str | None = None,
    warmup_steps: int = 0,
    pos_weight: float | None = None,
    patience: int = 5,
    logdir: Path | None = None,
) -> float:
    device, use_amp = setup_env()
    required = {"pitch", "velocity", "duration", "pos", "boundary", "bar"}
    train_rows = load_csv_rows(train_csv, required)
    val_rows = load_csv_rows(val_csv, required)
    section_vals = {r["section"] for r in train_rows + val_rows if "section" in r}
    mood_vals = {r["mood"] for r in train_rows + val_rows if "mood" in r}
    section_vocab = {s: i + 1 for i, s in enumerate(sorted(section_vals))} if section_vals else None
    mood_vocab = {s: i + 1 for i, s in enumerate(sorted(mood_vals))} if mood_vals else None
    ds_train = PhraseDataset(train_rows, max_len, section_vocab, mood_vocab)
    ds_val = PhraseDataset(val_rows, max_len, section_vocab, mood_vocab)
    pin_mem = pin_memory and device.type == "cuda"
    dl_train = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_mem,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_mem,
    )
    if arch == "lstm":
        model: nn.Module = PhraseLSTM(
            d_model=d_model,
            max_len=max_len,
            section_vocab_size=len(section_vocab) + 1 if section_vocab else 0,
            mood_vocab_size=len(mood_vocab) + 1 if mood_vocab else 0,
        )
    else:
        model = PhraseTransformer(
            d_model=d_model,
            max_len=max_len,
            section_vocab_size=len(section_vocab) + 1 if section_vocab else 0,
            mood_vocab_size=len(mood_vocab) + 1 if mood_vocab else 0,
        )
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = epochs * max(1, len(dl_train))
    sched = (
        torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
        if scheduler == "cosine"
        else None
    )
    pw = torch.tensor(pos_weight) if pos_weight else None
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    writer = SummaryWriter(logdir) if logdir and SummaryWriter else None

    def evaluate() -> float:
        model.eval()
        preds: list[int] = []
        trues: list[int] = []
        with torch.no_grad():
            for feats, y, mask in dl_val:
                feats = {k: v.to(device) for k, v in feats.items()}
                y = y.to(device)
                mask = mask.to(device)
                logits = model(feats, mask)
                p = (torch.sigmoid(logits[mask]) > 0.5).int()
                preds.extend(p.cpu().tolist())
                trues.extend(y[mask].int().cpu().tolist())
        return float(f1_score(trues, preds))

    best_f1, best_state = -1.0, None
    bad_epochs = 0
    global_step = 0
    for ep in range(epochs):
        t0 = time.time()
        model.train()
        for feats, y, mask in dl_train:
            feats = {k: v.to(device) for k, v in feats.items()}
            y = y.to(device)
            mask = mask.to(device)
            opt.zero_grad()
            ctx = (
                torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp)
                if use_amp
                else nullcontext()
            )
            with ctx:
                logits = model(feats, mask)
                loss = crit(logits[mask], y[mask])
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()
            if sched:
                if global_step < warmup_steps:
                    lr_scale = float(global_step + 1) / warmup_steps
                    for pg in opt.param_groups:
                        pg["lr"] = lr * lr_scale
                else:
                    sched.step()
            global_step += 1
        f1 = evaluate()
        logging.info("epoch %d f1 %.3f (%.1fs)", ep + 1, f1, time.time() - t0)
        if writer:
            writer.add_scalar("val/f1", f1, ep)
            writer.add_scalar("train/loss", float(loss), ep)
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    if best_state is not None:
        torch.save(best_state, out)
    if writer:
        writer.close()
    metrics_path = out.parent / "metrics.json"
    metrics_path.write_text(json.dumps({"f1": best_f1}))
    print(json.dumps({"f1": best_f1}))
    with torch.no_grad():
        feats, _, mask = next(iter(dl_val))
        feats = {k: v.to(device) for k, v in feats.items()}
        mask = mask.to(device)
        sample = model(feats, mask)
        print("inference", list(sample.shape), sample.flatten()[:3].cpu().tolist())
    return best_f1, device.type


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("train_csv", type=Path)
    parser.add_argument("val_csv", type=Path)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--out", type=Path, default=Path("phrase.ckpt"))
    parser.add_argument(
        "--arch", choices=["transformer", "lstm"], default="transformer"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--scheduler", choices=["cosine"], default=None)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--pos-weight", type=float)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--logdir", type=Path)
    parser.add_argument("--min-f1", type=float, default=-1.0)
    args = parser.parse_args(argv)

    f1, device_type = train_model(
        args.train_csv,
        args.val_csv,
        args.epochs,
        args.arch,
        args.out,
        batch_size=args.batch_size,
        d_model=args.d_model,
        max_len=args.max_len,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        grad_clip=args.grad_clip,
        lr=args.lr,
        weight_decay=args.weight_decay,
        scheduler=args.scheduler,
        warmup_steps=args.warmup_steps,
        pos_weight=args.pos_weight,
        patience=args.patience,
        logdir=args.logdir,
    )
    hparams = vars(args).copy()
    hparams["device"] = device_type
    (args.out.parent / "hparams.json").write_text(json.dumps(hparams))
    return 0 if args.min_f1 < 0 or f1 >= args.min_f1 else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
