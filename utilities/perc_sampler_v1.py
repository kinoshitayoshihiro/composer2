from __future__ import annotations
from collections import defaultdict, Counter
from dataclasses import dataclass
from pathlib import Path
from random import Random
import pickle
import math
import pretty_midi

from utilities.groove_sampler import infer_resolution

PERC_NOTE_MAP = {
    39: "conga_cl",
    40: "conga_op",
    70: "shaker",
    71: "shaker",
    72: "tamb",
    73: "tamb",
    74: "clap",
    75: "clap",
    76: "cowbell",
    77: "cowbell",
}

@dataclass
class PercModel:
    order: int
    resolution: int
    freq: dict[tuple[str, ...], Counter[str]]


def _token(step: int, label: str, vel: int) -> str:
    suf = "_h" if vel >= 64 else "_s"
    return f"{label}@{step}{suf}"


def _parse_midi(path: Path) -> tuple[list[str], list[float]]:
    pm = pretty_midi.PrettyMIDI(str(path))
    tempo = pm.get_tempo_changes()[0]
    bpm = float(tempo[0]) if tempo.size else 120.0
    sec_per_beat = 60.0 / bpm
    beats: list[float] = []
    events: list[tuple[float, int, int]] = []
    for inst in pm.instruments:
        for n in inst.notes:
            lbl = PERC_NOTE_MAP.get(n.pitch)
            if lbl:
                beat = n.start / sec_per_beat
                beats.append(beat)
                events.append((beat, n.velocity, n.pitch))
    return events, beats


def train(loop_dir: Path, *, order: int = 4, auto_res: bool = False) -> PercModel:
    events_all: list[list[str]] = []
    beats: list[float] = []
    for p in loop_dir.glob("*.mid*"):
        ev, b = _parse_midi(p)
        if not ev:
            continue
        beats.extend(b)
        events_all.append(ev)
    if not events_all:
        raise ValueError("no events")
    res = infer_resolution(beats) if auto_res else 16
    freq: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
    for ev in events_all:
        tokens = []
        for beat, vel, pitch in ev:
            step = int(round(beat * res / 4)) % res
            tokens.append(_token(step, PERC_NOTE_MAP[pitch], vel))
        hist = ["<s>"] * (order - 1) + tokens + ["<e>"]
        for i in range(order - 1, len(hist)):
            ctx = tuple(hist[i - order + 1 : i])
            nxt = hist[i]
            freq[ctx][nxt] += 1
    return PercModel(order, res, freq)


def save(model: PercModel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(model, fh)


def load(path: Path) -> PercModel:
    with path.open("rb") as fh:
        return pickle.load(fh)


def _sample_next(model: PercModel, ctx: list[str], rng: Random, temperature: float) -> str:
    for l in range(len(ctx), -1, -1):
        sub = tuple(ctx[-l:])
        dist = model.freq.get(sub)
        if dist:
            break
    else:
        dist = Counter()
    if not dist:
        dist = model.freq.get((), Counter())
    items = list(dist.items())
    probs = [c for _, c in items]
    total = sum(probs)
    probs = [p / total for p in probs]
    if temperature != 1.0:
        probs = [math.pow(p, 1.0 / temperature) for p in probs]
        s = sum(probs)
        probs = [p / s for p in probs]
    choices = [it[0] for it in items]
    return rng.choices(choices, probs)[0]


def generate_bar(history: list[str], *, model: PercModel, temperature: float = 1.0, seed: int | None = None) -> list[dict]:
    rng = Random(seed)
    res = model.resolution
    ctx = history[-(model.order - 1) :] if model.order > 1 else []
    tokens: list[str] = []
    for _ in range(res):
        tok = _sample_next(model, ctx, rng, temperature)
        tokens.append(tok)
        ctx.append(tok)
        if len(ctx) > model.order - 1:
            ctx.pop(0)
    history[:] = ctx
    events: list[dict] = []
    for tok in tokens:
        if tok in {"<s>", "<e>"}:
            continue
        lbl_step, suf = tok.rsplit("_", 1)
        label, step_str = lbl_step.split("@")
        step = int(step_str)
        velocity = 80 if suf == "h" else 50
        events.append({
            "instrument": label,
            "offset": step / res,
            "duration": 1 / res,
            "velocity": velocity,
        })
    return events


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["train", "sample"])
    ap.add_argument("path", type=Path)
    ap.add_argument("-o", "--output", type=Path, default=Path("perc.pkl"))
    ap.add_argument("--order", type=int, default=4)
    ap.add_argument("--auto-res", action="store_true")
    ap.add_argument("--length", type=int, default=4)
    ns = ap.parse_args()
    if ns.command == "train":
        model = train(ns.path, order=ns.order, auto_res=ns.auto_res)
        save(model, ns.output)
    else:
        model = load(ns.path)
        hist: list[str] = []
        ev = generate_bar(hist, model=model, seed=42)
        print(ev)

