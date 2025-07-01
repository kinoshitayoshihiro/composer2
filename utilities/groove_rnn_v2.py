from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click
import sys

try:
    import pytorch_lightning as pl
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except Exception as exc:  # pragma: no cover - optional dependency
    if "pytest" in sys.modules:
        pl = None  # type: ignore[assignment]
        torch = None  # type: ignore[assignment]
        nn = object  # type: ignore[assignment]
        DataLoader = object  # type: ignore[assignment]
        Dataset = object  # type: ignore[assignment]
    else:
        raise RuntimeError("Install extras: rnn") from exc

from .groove_sampler_ngram import RESOLUTION, Event

_VEL_BINS = 8
_MICRO_BINS = 8


def _bucket_vel(v: int) -> int:
    return min(_VEL_BINS - 1, max(0, v * _VEL_BINS // 128))


def _bucket_micro(m: int) -> int:
    return min(_MICRO_BINS - 1, max(0, (m + 32) * _MICRO_BINS // 64))


class TokenDataset(Dataset):
    def __init__(self, loops: list[list[tuple[int, str, int, int]]]) -> None:
        self._vocab: dict[tuple[int, str], int] = {}
        self._tokens: list[tuple[int, int, int]] = []
        for entry in loops:
            for step, lbl, vel, micro in entry:
                idx = self._vocab.setdefault((step, lbl), len(self._vocab))
                self._tokens.append((idx, _bucket_vel(vel), _bucket_micro(micro)))

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    @property
    def vocab(self) -> dict[tuple[int, str], int]:
        return self._vocab

    def __len__(self) -> int:  # type: ignore[override]
        return len(self._tokens)

    def __getitem__(self, idx: int) -> tuple[int, int, int]:  # type: ignore[override]
        return self._tokens[idx]


class GrooveRNN(pl.LightningModule):
    def __init__(self, vocab: int, embed: int = 64, hidden: int = 256) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.embed = nn.Embedding(vocab, embed)
        self.vel_emb = nn.Embedding(_VEL_BINS, 8)
        self.micro_emb = nn.Embedding(_MICRO_BINS, 8)
        self.gru = nn.GRU(embed + 16, hidden, num_layers=3, batch_first=True)
        self.token_head = nn.Linear(hidden, vocab)
        self.vel_head = nn.Linear(hidden, _VEL_BINS)
        self.tick_head = nn.Linear(hidden, _MICRO_BINS)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self._teacher_forcing = 1.0

    def configure_optimizers(self) -> Any:
        return torch.optim.Adam(self.parameters(), lr=self.hparams.get("lr", 1e-3))

    def forward(
        self, tok: torch.Tensor, vel: torch.Tensor, micro: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        emb = torch.cat([self.embed(tok), self.vel_emb(vel), self.micro_emb(micro)], dim=-1)
        out, _ = self.gru(emb)
        tok_logits = self.token_head(out)
        vel_logits = self.vel_head(out)
        tick_logits = self.tick_head(out)
        return tok_logits, vel_logits, tick_logits

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        idx, vel, micro = batch
        tok_logits, v_logits, t_logits = self(idx, vel, micro)
        loss1 = nn.functional.cross_entropy(tok_logits.view(-1, tok_logits.size(-1)), idx.view(-1))
        loss2 = nn.functional.cross_entropy(v_logits.view(-1, _VEL_BINS), vel.view(-1))
        loss3 = nn.functional.cross_entropy(t_logits.view(-1, _MICRO_BINS), micro.view(-1))
        return loss1 + loss2 + loss3


@dataclass
class TrainParams:
    epochs: int = 10
    lr: float = 1e-3
    batch_size: int = 32
    embed: int = 64
    hidden: int = 256


def _load_loops(path: Path) -> list[list[tuple[int, str, int, int]]]:
    import json

    with path.open("r", encoding="utf-8") as fh:
        obj = json.load(fh)
    loops = [list(map(tuple, entry["tokens"])) for entry in obj["data"]]
    return loops


def train_rnn_v2(
    loop_dir: Path,
    epochs: int = 10,
    lr: float = 1e-3,
    batch: int = 32,
    progress: bool = True,
    optuna_trials: int = 0,
    auto_tag: bool = False,
) -> tuple[GrooveRNN, dict]:
    loops = _load_loops(loop_dir)
    ds = TokenDataset(loops)
    dl = DataLoader(ds, batch_size=batch, shuffle=True)
    model = GrooveRNN(ds.vocab_size, embed=64, hidden=256)
    trainer = pl.Trainer(
        max_epochs=epochs,
        logger=False,
        enable_model_summary=False,
        enable_progress_bar=progress,
    )
    trainer.fit(model, dl)
    meta = {"vocab": ds.vocab}
    if auto_tag:
        from data_ops.auto_tag import auto_tag as _auto_tag

        meta["tags"] = _auto_tag(loop_dir)
    return model, meta


def sample_rnn_v2(
    model_ckpt: tuple[GrooveRNN, dict],
    bars: int = 4,
    temperature: float = 1.0,
    seed: int | None = None,
) -> list[Event]:
    model, meta = model_ckpt
    inv_vocab = {v: k for k, v in meta["vocab"].items()}
    tokens = [0]
    velocities = [0]
    micros = [0]
    model.eval()
    for _ in range(bars * RESOLUTION - 1):
        idx = torch.tensor(tokens[-1:])
        vel = torch.tensor(velocities[-1:])
        mic = torch.tensor(micros[-1:])
        with torch.no_grad():
            tok_logits, v_logits, t_logits = model(idx, vel, mic)
            tok_logits = tok_logits[0, -1]
            if temperature <= 0:
                next_idx = (tokens[-1] + 1) % len(inv_vocab)
            else:
                probs = torch.softmax(tok_logits / temperature, dim=-1)
                next_idx = int(torch.multinomial(probs, 1).item())
            v = int(torch.argmax(v_logits[0, -1]))
            m = int(torch.argmax(t_logits[0, -1]))
        tokens.append(next_idx)
        velocities.append(v)
        micros.append(m)
    events: list[Event] = []
    for i, t in enumerate(tokens):
        step, lbl = inv_vocab.get(t, (0, "kick"))
        bar_idx = i // RESOLUTION
        off = bar_idx * 4.0 + (step + micros[i] * 8 - 32) / (RESOLUTION / 4) / _MICRO_BINS
        vel = int(velocities[i] * 128 / _VEL_BINS)
        events.append({"instrument": lbl, "offset": off, "duration": 0.25, "velocity": vel})
    return events


@click.group()
def cli() -> None:
    """RNN v2 commands."""


@cli.command()
@click.argument("loops", type=Path)
@click.option("--epochs", type=int, default=10)
@click.option("--lr", type=float, default=1e-3)
@click.option("--batch", type=int, default=32)
@click.option("--out", "out_path", type=Path, required=True)
@click.option("--auto-tag/--no-auto-tag", default=False, help="Infer aux metadata")
def train_cmd(
    loops: Path, epochs: int, lr: float, batch: int, out_path: Path, auto_tag: bool
) -> None:
    model, meta = train_rnn_v2(
        loops, epochs=epochs, lr=lr, batch=batch, auto_tag=auto_tag
    )
    torch.save({"state_dict": model.state_dict(), "meta": meta}, out_path)
    click.echo(f"saved {out_path}")


@cli.command()
@click.argument("model_path", type=Path)
@click.option("-l", "--length", type=int, default=4)
@click.option("--temperature", type=float, default=1.0)
@click.option("--seed", type=int, default=42)
def sample_cmd(model_path: Path, length: int, temperature: float, seed: int) -> None:
    obj = torch.load(model_path, map_location="cpu")
    model = GrooveRNN(len(obj["meta"]["vocab"]))
    model.load_state_dict(obj["state_dict"])
    events = sample_rnn_v2((model, obj["meta"]), bars=length, temperature=temperature, seed=seed)
    click.echo(json.dumps(events))
