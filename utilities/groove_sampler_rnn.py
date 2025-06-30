"""Bidirectional GRU groove sampler."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Any, TypedDict

import click

try:  # optional dependency
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except Exception:  # pragma: no cover - optional dependency missing
    torch = None  # type: ignore
    nn = object  # type: ignore
    Dataset = object  # type: ignore
    DataLoader = object  # type: ignore

from .groove_sampler_ngram import DEFAULT_AUX, Event, RESOLUTION, PPQ, _hash_aux


class LoopEntry(TypedDict):
    tokens: list[tuple[int, str, int, int]]
    tempo_bpm: float
    bar_beats: int
    section: str | None
    heat_bin: int | None
    intensity: str | None


def _load_loops(path: Path) -> list[LoopEntry]:
    with path.open("r", encoding="utf-8") as fh:
        obj = json.load(fh)
    res: list[LoopEntry] = []
    for entry in obj["data"]:
        res.append(
            {
                "tokens": [tuple(t) for t in entry["tokens"]],
                "tempo_bpm": entry["tempo_bpm"],
                "bar_beats": entry["bar_beats"],
                "section": entry.get("section") or DEFAULT_AUX["section"],
                "heat_bin": entry.get("heat_bin") if entry.get("heat_bin") is not None else DEFAULT_AUX["heat_bin"],
                "intensity": entry.get("intensity") or DEFAULT_AUX["intensity"],
            }
        )
    return res


@dataclass
class TrainConfig:
    epochs: int = 6
    embed: int = 64
    hidden: int = 128


if torch is not None:

    _VEL_BINS = 8
    _MICRO_BINS = 8

    def _bucket_vel(v: int) -> int:
        return min(_VEL_BINS - 1, max(0, v * _VEL_BINS // 128))

    def _bucket_micro(m: int) -> int:
        return min(_MICRO_BINS - 1, max(0, (m + 32) * _MICRO_BINS // 64))

    class TokenDataset(Dataset):
        def __init__(self, data: list[LoopEntry]) -> None:
            self._data = data
            self._vocab: dict[tuple[int, str], int] = {}
            self._tokens: list[tuple[int, int, int]] = []
            self._build_vocab()
            for entry in data:
                for step, lbl, vel, micro in entry["tokens"]:
                    token = (step, lbl)
                    idx = self._vocab.setdefault(token, len(self._vocab))
                    self._tokens.append((idx, _bucket_vel(vel), _bucket_micro(micro)))

        def _build_vocab(self) -> None:
            for entry in self._data:
                for step, lbl, _v, _m in entry["tokens"]:
                    key = (step, lbl)
                    if key not in self._vocab:
                        self._vocab[key] = len(self._vocab)

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

    class GRUModel(nn.Module):
        def __init__(self, vocab: int, hidden: int, layers: int) -> None:
            super().__init__()
            self.embed = nn.Embedding(vocab, 64)
            self.vel_emb = nn.Embedding(_VEL_BINS, 8)
            self.micro_emb = nn.Embedding(_MICRO_BINS, 8)
            self.gru = nn.GRU(80, hidden, num_layers=layers, bidirectional=True, batch_first=True)
            self.fc = nn.Linear(hidden * 2, vocab)
            self.log_softmax = nn.LogSoftmax(dim=-1)

        def forward(self, tok: torch.Tensor, vel: torch.Tensor, micro: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            emb = torch.cat([self.embed(tok), self.vel_emb(vel), self.micro_emb(micro)], dim=-1)
            out, _ = self.gru(emb.unsqueeze(1))
            return self.log_softmax(self.fc(out))

    def train(
        path: Path,
        *,
        epochs: int = 6,
        embed: int = 64,
        hidden: int = 128,
    ) -> tuple[GRUModel, dict]:
        data = _load_loops(path)
        ds = TokenDataset(data)
        dl = DataLoader(ds, batch_size=32, shuffle=True)
        model = GRUModel(ds.vocab_size, hidden, 1)
        opt = torch.optim.Adam(model.parameters(), lr=2e-3)
        model.train()
        for _ in range(epochs):
            for idx, vel, micro in dl:
                idx = idx.long()
                vel = vel.long()
                micro = micro.long()
                opt.zero_grad()
                out = model(idx, vel, micro)
                loss = nn.functional.nll_loss(out.squeeze(1), idx)
                loss.backward()
                opt.step()
        aux_map: dict[int, int] = {}
        for entry in data:
            for step, lbl, _v, _m in entry["tokens"]:
                key = ds.vocab[(step, lbl)]
                aux = (
                    entry.get("section") or DEFAULT_AUX["section"],
                    str(entry.get("heat_bin") or DEFAULT_AUX["heat_bin"]),
                    entry.get("intensity") or DEFAULT_AUX["intensity"],
                )
                aux_map[key] = _hash_aux(aux)
        meta = {"vocab": ds.vocab, "aux": aux_map}
        return model, meta

    def save(model: GRUModel, meta: dict, path: Path) -> None:
        torch.save({"state": model.state_dict(), "meta": meta}, path)

    def load(path: Path) -> tuple[GRUModel, dict]:
        obj = torch.load(path, map_location="cpu")
        meta = obj["meta"]
        model = GRUModel(len(meta["vocab"]), 128, 2)
        model.load_state_dict(obj["state"])
        model.eval()
        return model, meta

    def sample(
        model: GRUModel,
        meta: dict,
        *,
        bars: int = 4,
        temperature: float = 1.0,
        humanize: bool = False,
        rng: Random | None = None,
    ) -> list[Event]:
        rng = rng or Random()
        inv_vocab = {v: k for k, v in meta["vocab"].items()}
        tokens = [rng.randrange(len(inv_vocab))]
        model.eval()
        for _ in range(bars * RESOLUTION - 1):
            inp = torch.tensor(tokens[-1:])
            with torch.no_grad():
                out = model(inp, torch.zeros(1, dtype=torch.long), torch.zeros(1, dtype=torch.long))
                logits = out[0, -1]
                if temperature <= 0:
                    idx = int(torch.argmax(logits).item())
                else:
                    probs = torch.exp(logits / temperature)
                    idx = int(torch.multinomial(probs, 1).item())
            tokens.append(idx)
        events: list[Event] = []
        for i, t in enumerate(tokens):
            step, lbl = inv_vocab[t]
            bar_idx = i // RESOLUTION
            vel = 100
            micro = 0
            if humanize:
                vel = int(rng.gauss(100.0, 6.0))
                vel = max(1, min(127, vel))
                micro = int(rng.gauss(0.0, 12.0))
                micro = max(-45, min(45, micro))
            offset = bar_idx * 4.0 + (step + micro / PPQ) / (RESOLUTION / 4)
            events.append(
                {
                    "instrument": lbl,
                    "offset": offset,
                    "duration": 0.25,
                    "velocity": vel,
                }
            )
        return events
else:
    import json
    from dataclasses import dataclass
    import numpy as np

    @dataclass
    class GRUModel:
        """Minimal fallback model using bigram probabilities."""

        matrix: np.ndarray

    class TokenDataset:
        def __init__(self, data: list[LoopEntry]) -> None:
            self.tokens: list[tuple[int, str]] = []
            self.vocab: dict[tuple[int, str], int] = {}
            for entry in data:
                for step, lbl, _v, _m in entry["tokens"]:
                    token = (step, lbl)
                    idx = self.vocab.setdefault(token, len(self.vocab))
                    self.tokens.append(idx)

        @property
        def vocab_size(self) -> int:
            return len(self.vocab)

    def train(path: Path, *, epochs: int = 1, **_: object) -> tuple[GRUModel, dict]:
        data = _load_loops(path)
        ds = TokenDataset(data)
        mat = np.ones((ds.vocab_size, ds.vocab_size), dtype=float)
        for a, b in zip(ds.tokens[:-1], ds.tokens[1:]):
            mat[a, b] += 1.0
        mat /= mat.sum(axis=1, keepdims=True)
        model = GRUModel(mat)
        meta = {"vocab": ds.vocab}
        return model, meta

    def save(model: GRUModel, meta: dict, path: Path) -> None:
        data = {"matrix": model.matrix.tolist(), "meta": meta}
        path.write_text(json.dumps(data))

    def load(path: Path) -> tuple[GRUModel, dict]:
        obj = json.loads(path.read_text())
        mat = np.array(obj["matrix"], dtype=float)
        model = GRUModel(mat)
        return model, obj["meta"]

    def sample(
        model: GRUModel,
        meta: dict,
        *,
        bars: int = 4,
        temperature: float = 1.0,
        humanize: bool = False,
        rng: Random | None = None,
    ) -> list[Event]:
        rng = rng or Random()
        inv_vocab = {v: k for k, v in meta["vocab"].items()}
        idx = rng.randrange(len(inv_vocab))
        events: list[Event] = []
        for i in range(bars * RESOLUTION):
            step, lbl = inv_vocab[idx]
            offset = i / (RESOLUTION / 4)
            vel = 100
            if humanize:
                vel = max(1, min(127, int(rng.gauss(100.0, 6.0))))
            events.append({"instrument": lbl, "offset": offset, "duration": 0.25, "velocity": vel})
            probs = model.matrix[idx]
            if temperature <= 0:
                idx = int(np.argmax(probs))
            else:
                probs = np.exp(np.log(probs) / temperature)
                probs /= probs.sum()
                idx = int(rng.choices(range(len(probs)), weights=probs, k=1)[0])
        return events


@click.group()
def cli() -> None:
    """RNN groove sampler commands."""


@cli.command()
@click.argument("loops", type=Path)
@click.option("--epochs", default=6, type=int)
@click.option("--embed", default=8, type=int)
@click.option("--hidden", default=16, type=int)
@click.option("--out", "out_path", type=Path, required=True)
def train_cmd(
    loops: Path, epochs: int, embed: int, hidden: int, out_path: Path
) -> None:
    model, meta = train(loops, epochs=epochs, embed=embed, hidden=hidden)
    save(model, meta, out_path)
    click.echo(f"saved model to {out_path}")


@cli.command()
@click.argument("model_path", type=Path)
@click.option("-l", "--length", default=4, type=int)
@click.option("--temperature", default=1.0, type=float)
@click.option("--humanize/--no-humanize", default=False)
@click.option("-o", "--out", type=Path)
def sample_cmd(
    model_path: Path,
    length: int,
    temperature: float,
    humanize: bool,
    out: Path | None,
) -> None:
    model, meta = load(model_path)
    events = sample(
        model,
        meta,
        bars=length,
        temperature=temperature,
        humanize=humanize,
    )
    if out is None:
        click.echo(json.dumps(events))
    else:
        import pretty_midi

        pm = pretty_midi.PrettyMIDI(initial_tempo=120)
        inst = pretty_midi.Instrument(program=0, is_drum=True)
        for ev in events:
            inst.notes.append(
                pretty_midi.Note(
                    velocity=ev["velocity"],
                    pitch=36 if ev["instrument"] == "kick" else 38,
                    start=ev["offset"],
                    end=ev["offset"] + ev["duration"],
                )
            )
        pm.instruments.append(inst)
        pm.write(str(out))
        click.echo(f"wrote {out}")

