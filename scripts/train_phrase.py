from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import subprocess
import sys
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable


# local import guard so the script works without an editable install
try:  # pragma: no cover - import robustness
    from models.phrase_transformer import PhraseTransformer
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parent.parent
    if os.environ.get("ALLOW_LOCAL_IMPORT") == "1":
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
            logging.warning("ALLOW_LOCAL_IMPORT=1, inserted repo root into sys.path")
        try:
            from models.phrase_transformer import PhraseTransformer
        except ModuleNotFoundError as exc:  # pragma: no cover - still failing
            print(
                "Could not import project modules. Run 'pip install -e .' or set PYTHONPATH",
                file=sys.stderr,
            )
            raise SystemExit(1) from exc
    else:  # pragma: no cover - guidance when fallback disabled
        print(
            "Could not import project modules. Run 'pip install -e .' or set PYTHONPATH",
            file=sys.stderr,
        )
        raise SystemExit(1)


FIELDS = [
    "pitch",
    "velocity",
    "duration",
    "pos",
    "boundary",
    "bar",
    "instrument",
    "section",
    "mood",
]


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


def write_csv(rows: Iterable[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def load_corpus(
    root: Path,
    *,
    filter_section: str | None = None,
    filter_mood: str | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Load JSONL corpus with train/valid splits and optional tag filtering."""

    def load_split(split: str) -> list[dict[str, object]]:
        base = root / split
        files: list[Path] = []
        if (base / "samples.jsonl").is_file():
            files = [base / "samples.jsonl"]
        else:
            files = sorted((base / "samples").glob("*.jsonl"))
        rows: list[dict[str, object]] = []
        for p in files:
            with p.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    tags = obj.get("tags", {})
                    if filter_section and tags.get("section") != filter_section:
                        continue
                    if filter_mood and tags.get("mood") != filter_mood:
                        continue
                    rows.append(
                        {
                            "pitch": obj["pitch"],
                            "velocity": obj.get("velocity", 0),
                            "duration": obj.get("duration", 0),
                            "pos": obj.get("pos", 0),
                            "boundary": obj.get("boundary", 0),
                            "bar": obj.get("bar", 0),
                            "instrument": obj.get("instrument", ""),
                            "section": tags.get("section", ""),
                            "mood": tags.get("mood", ""),
                        }
                    )
        return rows

    return load_split("train"), load_split("valid")


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
    auto_pos_weight: bool = False,
    resume: Path | None = None,
    save_every: int = 0,
    early_stopping: int = 0,
    f1_scan_range: tuple[float, float, float] = (0.2, 0.8, 0.1),
    logdir: Path | None = None,
    precision: str | None = None,
) -> tuple[float, str]:
    """Train the phrase boundary model and return the best F1 and device."""

    import importlib

    try:  # optional dependency
        import numpy as np  # type: ignore
    except Exception:  # pragma: no cover - numpy missing
        np = None  # type: ignore

    torch = importlib.import_module("torch")
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    try:  # optional
        from torch.utils.tensorboard import SummaryWriter
    except Exception:  # pragma: no cover
        SummaryWriter = None

    def setup_env(seed: int = 42) -> tuple[torch.device, bool]:
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
        if device.type == "mps":
            amp = False
            torch.set_float32_matmul_precision(precision or "medium")
        elif precision:
            torch.set_float32_matmul_precision(precision)
        if device.type == "cuda":
            try:
                torch.use_deterministic_algorithms(True)
            except Exception as exc:  # pragma: no cover
                logging.warning("deterministic algos unavailable: %s", exc)
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
            vel = torch.tensor(
                [float(r["velocity"]) for r in g] + [0] * pad, dtype=torch.float32
            )
            dur = torch.tensor(
                [float(r["duration"]) for r in g] + [0] * pad, dtype=torch.float32
            )
            pos = torch.tensor([int(r["pos"]) for r in g] + [0] * pad, dtype=torch.long)
            y = torch.tensor(
                [float(r["boundary"]) for r in g] + [0] * pad, dtype=torch.float32
            )
            feats = {
                "pitch_class": pc,
                "velocity": vel,
                "duration": dur,
                "position": pos,
            }
            if self.section_vocab:
                sec = [
                    self.section_vocab.get(r.get("section", ""), 0) for r in g
                ] + [0] * pad
                feats["section"] = torch.tensor(sec, dtype=torch.long)
            if self.mood_vocab:
                md = [self.mood_vocab.get(r.get("mood", ""), 0) for r in g] + [
                    0
                ] * pad
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

    device, use_amp = setup_env()
    logging.info(
        "device %s amp=%s precision=%s",
        device,
        use_amp,
        torch.get_float32_matmul_precision(),
    )

    required = {"pitch", "velocity", "duration", "pos", "boundary", "bar"}
    train_rows = load_csv_rows(train_csv, required)
    val_rows = load_csv_rows(val_csv, required)
    section_vals = {r["section"] for r in train_rows + val_rows if "section" in r}
    mood_vals = {r["mood"] for r in train_rows + val_rows if "mood" in r}
    if auto_pos_weight and pos_weight is None:
        positive_count = sum(int(r["boundary"]) for r in train_rows)
        total_count = len(train_rows)
        p = max(1e-6, min(1 - 1e-6, positive_count / max(1, total_count)))
        pos_weight = (1 - p) / p
        logging.info("auto pos_weight=%s", pos_weight)
    section_vocab = (
        {s: i + 1 for i, s in enumerate(sorted(section_vals))} if section_vals else None
    )
    mood_vocab = (
        {s: i + 1 for i, s in enumerate(sorted(mood_vals))} if mood_vals else None
    )
    ds_train = PhraseDataset(train_rows, max_len, section_vocab, mood_vocab)
    ds_val = PhraseDataset(val_rows, max_len, section_vocab, mood_vocab)

    pin_mem = device.type == "cuda" or pin_memory
    persist = device.type == "cuda" and num_workers > 0

    def worker_init_fn(worker_id: int) -> None:
        seed = torch.initial_seed() % 2**32
        torch.manual_seed(seed + worker_id)
        random.seed(seed + worker_id)
        if np is not None:
            np.random.seed(seed + worker_id)

    dl_train = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_mem,
        persistent_workers=persist,
        worker_init_fn=worker_init_fn if num_workers else None,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_mem,
        persistent_workers=persist,
        worker_init_fn=worker_init_fn if num_workers else None,
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
    pw = torch.tensor([pos_weight], device=device) if pos_weight else None
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    writer = SummaryWriter(logdir) if logdir and SummaryWriter else None

    start_epoch = 0
    global_step = 0
    best_f1 = -1.0
    if resume and resume.is_file():
        state = torch.load(resume, map_location="cpu")
        model.load_state_dict(state["model"])
        opt.load_state_dict(state.get("optimizer", {}))
        if sched and state.get("scheduler"):
            sched.load_state_dict(state["scheduler"])
        start_epoch = int(state.get("epoch", 0))
        global_step = int(state.get("global_step", 0))
        best_f1 = float(state.get("best_f1", -1.0))

    def evaluate() -> tuple[float, float]:
        model.eval()
        probs: list[float] = []
        trues: list[int] = []
        with torch.no_grad():
            for feats, y, mask in dl_val:
                feats = {k: v.to(device) for k, v in feats.items()}
                y = y.to(device)
                mask = mask.to(device)
                logits = model(feats, mask)
                probs.extend(torch.sigmoid(logits[mask]).cpu().tolist())
                trues.extend(y[mask].int().cpu().tolist())
        best_f1, best_th = -1.0, 0.5
        start, end, step = f1_scan_range
        if np is not None:
            ths = np.arange(start, end + 1e-9, step)
        else:
            ths = []
            t = start
            while t <= end + 1e-9:
                ths.append(t)
                t += step
        for th in ths:
            preds = [1 if p > th else 0 for p in probs]
            f1 = f1_score(trues, preds)
            if f1 > best_f1:
                best_f1, best_th = f1, float(th)
        return best_f1, best_th

    best_state = None
    best_threshold = 0.5
    bad_epochs = 0
    metrics_rows: list[dict[str, float]] = []
    for ep in range(start_epoch, epochs):
        t0 = time.time()
        model.train()
        loss_sum = 0.0
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
            loss_sum += float(loss)
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
        f1, th = evaluate()
        avg_loss = loss_sum / max(1, len(dl_train))
        lr_cur = opt.param_groups[0]["lr"]
        elapsed = time.time() - t0
        logging.info(
            "epoch %d train_loss %.4f val_f1 %.3f lr %.2e th %.2f time %.1fs",
            ep + 1,
            avg_loss,
            f1,
            lr_cur,
            th,
            elapsed,
        )
        metrics_rows.append(
            {
                "epoch": ep + 1,
                "loss": avg_loss,
                "f1": f1,
                "best_th": th,
                "time": elapsed,
            }
        )
        if writer:
            writer.add_scalar("val/f1", f1, ep)
            writer.add_scalar("train/loss", avg_loss, ep)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = th
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if early_stopping and bad_epochs >= early_stopping:
                break
        if save_every and (ep + 1) % save_every == 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "scheduler": sched.state_dict() if sched else None,
                    "epoch": ep + 1,
                    "global_step": global_step,
                    "best_f1": best_f1,
                },
                out.with_suffix(f".epoch{ep + 1}.ckpt"),
            )
    final_state = {
        "model": {k: v.cpu() for k, v in model.state_dict().items()},
        "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict() if sched else None,
        "epoch": ep + 1,
        "global_step": global_step,
        "best_f1": best_f1,
    }
    torch.save(final_state, out)
    if best_state is not None:
        best_ckpt = final_state.copy()
        best_ckpt["model"] = best_state
        torch.save(best_ckpt, out.with_suffix(".best.ckpt"))
    if writer:
        writer.close()
    metrics_path = out.parent / "metrics.json"
    metrics_path.write_text(
        json.dumps({"f1": best_f1, "best_threshold": best_threshold}, ensure_ascii=False)
    )
    metrics_csv = out.parent / "metrics_epoch.csv"
    with metrics_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "loss", "f1", "best_th", "time"])
        writer.writeheader()
        writer.writerows(metrics_rows)
    preview_path = out.parent / "preds_preview.json"
    try:
        with torch.no_grad():
            feats, y, mask = next(iter(dl_val))
            feats = {k: v.to(device) for k, v in feats.items()}
            mask0 = mask[0].bool()
            logits = model(feats, mask.to(device))[0, mask0.to(device)]
            probs = torch.sigmoid(logits).cpu().tolist()
            preds = [1 if p > best_threshold else 0 for p in probs]
            trues = y[0][mask0].int().cpu().tolist()
        preview_path.write_text(
            json.dumps({"probs": probs, "preds": preds, "trues": trues}, ensure_ascii=False)
        )
    except StopIteration:  # pragma: no cover - empty validation set
        pass
    print(json.dumps({"f1": best_f1}))
    return best_f1, device.type


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "train_csv",
        type=Path,
        nargs="?",
        help="training CSV file (ignored when --data is used)",
    )
    parser.add_argument(
        "val_csv",
        type=Path,
        nargs="?",
        help="validation CSV file (ignored when --data is used)",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--out", type=Path, default=Path("phrase.ckpt"))
    parser.add_argument("--arch", choices=["transformer", "lstm"], default="transformer")
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
    parser.add_argument("--auto-pos-weight", action="store_true")
    parser.add_argument(
        "--f1-scan-range",
        nargs=3,
        type=float,
        default=(0.2, 0.8, 0.1),
        metavar=("START", "END", "STEP"),
        help="threshold search range",
    )
    parser.add_argument("--logdir", type=Path)
    parser.add_argument("--min-f1", type=float, default=-1.0)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--early-stopping", type=int, default=0)
    parser.add_argument("--precision", choices=["high", "medium", "low"], default=None)
    parser.add_argument(
        "--data",
        type=Path,
        help="corpus directory with train/valid splits; overrides positional CSVs",
    )
    parser.add_argument("--instrument")
    parser.add_argument("--filter-section", type=str, default=None)
    parser.add_argument("--filter-mood", type=str, default=None)
    parser.add_argument("--sample", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.f1_scan_range = tuple(args.f1_scan_range)

    handlers = [logging.StreamHandler()]
    if args.logdir:
        args.logdir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(args.logdir / "train.log"))
    logging.basicConfig(level=logging.INFO, handlers=handlers)

    if args.data:
        try:
            train_rows, val_rows = load_corpus(
                args.data,
                filter_section=args.filter_section,
                filter_mood=args.filter_mood,
            )
            if args.instrument:
                inst = args.instrument.lower()
                train_rows = [
                    r for r in train_rows if inst in str(r.get("instrument", "")).lower()
                ]
                val_rows = [
                    r for r in val_rows if inst in str(r.get("instrument", "")).lower()
                ]
            if args.sample:
                rng = random.Random(0)
                train_rows = rng.sample(train_rows, min(args.sample, len(train_rows)))
                val_rows = rng.sample(val_rows, min(args.sample, len(val_rows)))
            with tempfile.TemporaryDirectory() as td:
                tmp = Path(td)
                train_csv = tmp / "train.csv"
                val_csv = tmp / "valid.csv"
                write_csv(train_rows, train_csv)
                write_csv(val_rows, val_csv)
                args.train_csv = train_csv
                args.val_csv = val_csv
        except Exception:
            with tempfile.TemporaryDirectory() as td:
                tmp = Path(td)
                train_csv = tmp / "train.csv"
                val_csv = tmp / "valid.csv"
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "tools.corpus_to_phrase_csv",
                        "--in",
                        str(args.data),
                        "--out-train",
                        str(train_csv),
                        "--out-valid",
                        str(val_csv),
                    ],
                    check=True,
                )
                args.train_csv = train_csv
                args.val_csv = val_csv

    if not args.train_csv or not args.train_csv.is_file():
        raise SystemExit(f"missing train_csv {args.train_csv}")
    if not args.val_csv or not args.val_csv.is_file():
        raise SystemExit(f"missing val_csv {args.val_csv}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    run_path = args.out.with_suffix(".run.json")
    run_cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    run_path.write_text(json.dumps(run_cfg, ensure_ascii=False, indent=2))

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
        auto_pos_weight=args.auto_pos_weight,
        resume=args.resume,
        save_every=args.save_every,
        early_stopping=args.early_stopping,
        f1_scan_range=args.f1_scan_range,
        logdir=args.logdir,
        precision=args.precision,
    )

    hparams = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    hparams["device"] = device_type
    (args.out.parent / "hparams.json").write_text(
        json.dumps(hparams, ensure_ascii=False, indent=2)
    )
    return 0 if args.min_f1 < 0 or f1 >= args.min_f1 else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())

