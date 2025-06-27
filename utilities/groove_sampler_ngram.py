"""Portable n-gram groove sampler."""

from __future__ import annotations

import hashlib
import io
import json
import pickle
import sys
from collections import Counter, defaultdict
from collections.abc import Iterator, Sequence
from pathlib import Path
from random import Random
from typing import Any, TypedDict

import click
import pretty_midi

from .drum_map_registry import GM_DRUM_MAP

PPQ = 480
RESOLUTION = 16
VERSION = 1
ALPHA = 0.1


# mapping from MIDI pitch to drum label
_PITCH_TO_LABEL: dict[int, str] = {val[1]: k for k, val in GM_DRUM_MAP.items()}

State = tuple[int, str]
HashKey = tuple[State | int, ...]


class Model(TypedDict):
    """Container for the n-gram model."""

    version: int
    resolution: int
    order: int
    freq: dict[int, dict[HashKey, Counter[State]]]
    prob: dict[int, dict[HashKey, dict[State, float]]]
    mean_velocity: dict[str, float]
    vel_deltas: dict[str, Counter[int]]
    micro_offsets: dict[str, Counter[int]]
    aux_dims: list[str]


def _iter_midi(path: Path) -> Iterator[tuple[float, str, int, int]]:
    """Yield quantised events from a MIDI file."""

    pm = pretty_midi.PrettyMIDI(str(path))
    tempo = pm.get_tempo_changes()[1]
    bpm = float(tempo[0]) if tempo.size else 120.0
    sec_per_beat = 60.0 / bpm

    for inst in pm.instruments:
        if not inst.is_drum:
            continue
        for note in inst.notes:
            beat = note.start / sec_per_beat
            step = int(round(beat * (RESOLUTION / 4)))
            q_pos = step / (RESOLUTION / 4)
            micro = int(round((beat - q_pos) * PPQ))
            label = _PITCH_TO_LABEL.get(note.pitch, str(note.pitch))
            yield step, label, note.velocity, micro


def _iter_wav(path: Path) -> Iterator[tuple[int, str, int, int]]:
    """Return onset positions from a WAV file using librosa."""

    try:
        import librosa
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("WAV support requires librosa") from exc

    y, sr = librosa.load(path, sr=None, mono=True)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(tempo) if tempo else 120.0
    sec_per_beat = 60.0 / bpm
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time")
    for t in onsets:
        beat = t / sec_per_beat
        step = int(round(beat * (RESOLUTION / 4)))
        micro = 0
        yield step, "perc", 100, micro


def _load_events(
    loop_dir: Path, exts: Sequence[str]
) -> tuple[list[tuple[list[State], str]], dict[str, float], dict[str, Counter[int]], dict[str, Counter[int]]]:
    events: list[tuple[list[State], str]] = []
    vel_map: dict[str, list[int]] = defaultdict(list)
    micro_map: dict[str, list[int]] = defaultdict(list)

    normalized = {e.replace("midi", "mid") for e in exts}
    for p in loop_dir.iterdir():
        suf = p.suffix.lower().lstrip(".")
        suf = "mid" if suf in {"mid", "midi"} else suf
        if suf not in normalized:
            continue
        seq: list[State] = []
        if p.suffix.lower() in {".mid", ".midi"}:
            iterator = _iter_midi(p)
        else:
            iterator = _iter_wav(p)
        for step, lbl, vel, micro in iterator:
            seq.append(((step % RESOLUTION), lbl))
            vel_map[lbl].append(vel)
            micro_map[lbl].append(micro)
        if seq:
            events.append((sorted(seq, key=lambda x: x[0]), p.name))
    mean_velocity = {k: sum(v) / len(v) for k, v in vel_map.items() if v}
    vel_deltas = {
        k: Counter(int(v - mean_velocity[k]) for v in vals)
        for k, vals in vel_map.items()
    }
    micro_offsets = {k: Counter(vals) for k, vals in micro_map.items()}
    return events, mean_velocity, vel_deltas, micro_offsets


DEFAULT_AUX = {"section": "verse", "heat_bin": 0, "intensity": "mid"}

_AUX_HASH_CACHE: dict[int, tuple[str, str, str]] = {}


def _hash_aux(aux: tuple[str, str, str]) -> int:
    data = "|".join(aux).encode("utf-8")
    val = int.from_bytes(hashlib.blake2s(data, digest_size=4).digest(), "big")
    prev = _AUX_HASH_CACHE.setdefault(val, aux)
    if prev != aux:
        raise RuntimeError(f"hash collision: {prev} vs {aux}")
    return val


def train(
    loop_dir: Path,
    *,
    ext: str = "midi",
    order: int | str = "auto",
    aux_map: dict[str, dict[str, Any]] | None = None,
) -> Model:
    """Train an n-gram model from MIDI/WAV loops."""

    exts = [e.strip().lower() for e in ext.split(",") if e]
    seqs, mean_vel, vel_deltas, micro_offsets = _load_events(loop_dir, exts)
    if not seqs:
        raise ValueError("no events found")

    if order == "auto":
        avg_len = sum(len(seq) for seq, _ in seqs) / len(seqs)
        n = 3 if avg_len >= 8 else 2
    else:
        n = int(order)

    aux_map = aux_map or {}

    freq: dict[int, dict[tuple[State, ...], Counter[State]]] = {
        i: defaultdict(Counter) for i in range(n)
    }
    aux_dims = ["section", "heat_bin", "intensity"]
    for seq, name in seqs:
        meta = aux_map.get(name, {})
        section = str(meta.get("section", DEFAULT_AUX["section"]))
        heat_bin = str(meta.get("heat_bin", DEFAULT_AUX["heat_bin"]))
        intensity = str(meta.get("intensity", DEFAULT_AUX["intensity"]))
        aux_full = _hash_aux((section, heat_bin, intensity))
        aux_any_int = _hash_aux((section, heat_bin, "*"))
        for s in seq:
            freq[0][(aux_full,)][s] += 1
            freq[0][(aux_any_int,)][s] += 1
            freq[0][()][s] += 1
        for i in range(1, n):
            for idx in range(len(seq) - i):
                ctx = tuple(seq[idx : idx + i])
                nxt = seq[idx + i]
                freq[i][ctx + (aux_full,)][nxt] += 1
                freq[i][ctx + (aux_any_int,)][nxt] += 1
                freq[i][ctx][nxt] += 1

    prob: dict[int, dict[tuple[State, ...], dict[State, float]]] = {i: {} for i in range(n)}
    for order_i, ctx_map in freq.items():
        for ctx, counter in ctx_map.items():
            total = sum(counter.values())
            if total == 0:
                continue
            v = len(counter)
            prob[order_i][ctx] = {
                s: (c + ALPHA) / (total + ALPHA * v) for s, c in counter.items()
            }

    model: Model = Model(
        version=VERSION,
        resolution=RESOLUTION,
        order=n,
        freq=freq,
        prob=prob,
        mean_velocity=mean_vel,
        vel_deltas=vel_deltas,
        micro_offsets=micro_offsets,
        aux_dims=aux_dims,
    )
    return model


def save(model: Model, path: Path) -> None:
    with path.open("wb") as fh:
        pickle.dump(dict(model), fh)


def load(path: Path) -> Model:
    with path.open("rb") as fh:
        data = pickle.load(fh)
    if (
        data.get("resolution") != RESOLUTION
        or data.get("version") != VERSION
    ):
        raise RuntimeError("incompatible model version")
    return Model(**data)


def _choose(probs: dict[State, float], rng: Random) -> State:
    states = list(probs.keys())
    weights = list(probs.values())
    return rng.choices(states, weights, k=1)[0]


def _sample_next(
    history: Sequence[State],
    model: Model,
    rng: Random,
    *,
    cond: dict[str, Any] | None = None,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> State:
    n = model["order"]
    aux_dims = model.get("aux_dims", [])
    def _aux_hash(with_intensity: bool = True) -> int:
        if not aux_dims:
            return 0
        section = str(cond.get("section", DEFAULT_AUX["section"])) if cond else DEFAULT_AUX["section"]
        heat_bin = str(cond.get("heat_bin", DEFAULT_AUX["heat_bin"])) if cond else str(DEFAULT_AUX["heat_bin"])
        intensity = str(cond.get("intensity", DEFAULT_AUX["intensity"])) if cond else DEFAULT_AUX["intensity"]
        if not with_intensity:
            intensity = "*"
        return _hash_aux((section, heat_bin, intensity))

    aux_full = _aux_hash(True)
    aux_any = _aux_hash(False)
    for order in range(min(len(history), n - 1), -1, -1):
        ctx = tuple(history[-order:])
        dist = model["prob"].get(order, {}).get(ctx + (aux_full,))
        if not dist:
            dist = model["prob"].get(order, {}).get(ctx + (aux_any,))
        if not dist:
            dist = model["prob"].get(order, {}).get(ctx)
        if dist:
            break
    if not dist:
        dist = model["prob"][0].get((), {})
    if not dist:
        raise RuntimeError("No probability mass")
    if top_k is not None:
        dist = dict(sorted(dist.items(), key=lambda x: x[1], reverse=True)[:top_k])
    if temperature != 1.0:
        dist = {k: v ** (1.0 / temperature) for k, v in dist.items()}
        total = sum(dist.values())
        dist = {k: v / total for k, v in dist.items()}
    return _choose(dist, rng)


def sample(
    model: Model,
    *,
    bars: int = 4,
    temperature: float = 1.0,
    top_k: int | None = None,
    seed: int | None = None,
    cond: dict[str, Any] | None = None,
) -> list[dict[str, float | str]]:
    """Generate events using the trained model."""

    rng = Random(seed)
    events: list[dict[str, float | str]] = []
    history: list[State] = []
    for bar in range(bars):
        next_bin = 0
        while next_bin < RESOLUTION:
            state = _sample_next(
                history,
                model,
                rng,
                temperature=temperature,
                top_k=top_k,
                cond=cond,
            )
            step, lbl = state
            if step < next_bin:
                step = next_bin
            offset_beats = (bar * RESOLUTION + step) / (RESOLUTION / 4)
            micro_choices = list(model["micro_offsets"].get(lbl, {0: 1}).elements())
            micro = rng.choice(micro_choices) if micro_choices else 0
            vel_mean = int(model["mean_velocity"].get(lbl, 100))
            delta_choices = list(model["vel_deltas"].get(lbl, {0: 1}).elements())
            vel = int(vel_mean + (rng.choice(delta_choices) if delta_choices else 0))
            events.append(
                {
                    "instrument": lbl,
                    "offset": offset_beats + micro / PPQ,
                    "duration": 0.25,
                    "velocity": max(1, min(127, vel)),
                }
            )
            history.append((step, lbl))
            if len(history) > model["order"] - 1:
                history.pop(0)
            next_bin = step + 1
    events.sort(key=lambda x: x["offset"])
    return events


def events_to_midi(events: Sequence[dict[str, float | str]]) -> pretty_midi.PrettyMIDI:
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=0, is_drum=True)
    pitch_map = {k: v[1] for k, v in GM_DRUM_MAP.items()}
    for ev in events:
        start = float(ev.get("offset", 0.0)) * 0.5  # beat to seconds at 120bpm
        end = start + float(ev.get("duration", 0.25)) * 0.5
        pitch = pitch_map.get(str(ev.get("instrument")), 36)
        velocity = int(ev.get("velocity", 100))
        inst.notes.append(pretty_midi.Note(velocity=velocity, pitch=pitch, start=start, end=end))
    pm.instruments.append(inst)
    return pm


@click.group()
def cli() -> None:
    """Groove sampler commands."""


@cli.command()
@click.argument("loop_dir", type=Path)
@click.option("--ext", default="wav,midi", help="Comma separated extensions")
@click.option("--order", default="auto", help="n-gram order or 'auto'")
@click.option("--out", "out_path", type=Path, default=Path("model.pkl"))
@click.option(
    "--aux",
    "aux_path",
    type=Path,
    default=None,
    help="JSON map of loop names to aux data, e.g. '{\"foo.mid\": {\"section\": \"chorus\"}}'",
)
def train_cmd(loop_dir: Path, ext: str, order: str, out_path: Path, aux_path: Path | None) -> None:
    """Train a model from loops."""

    aux_map = None
    if aux_path and aux_path.exists():
        with aux_path.open("r", encoding="utf-8") as fh:
            aux_map = json.load(fh)
    model = train(loop_dir, ext=ext, order=order, aux_map=aux_map)
    save(model, out_path)
    click.echo(f"saved model to {out_path}")


@cli.command()
@click.argument("model_path", type=Path)
@click.option("-l", "--length", default=4, type=int)
@click.option("--temperature", default=1.0, type=float)
@click.option("--seed", default=42, type=int)
@click.option(
    "--cond",
    default=None,
    help="JSON aux condition, e.g. '{\"section\":\"chorus\"}'",
)
def sample_cmd(model_path: Path, length: int, temperature: float, seed: int, cond: str | None) -> None:
    """Generate MIDI from a model."""

    model = load(model_path)
    cond_map = json.loads(cond) if cond else None
    ev = sample(model, bars=length, temperature=temperature, seed=seed, cond=cond_map)
    pm = events_to_midi(ev)
    buf = io.BytesIO()
    pm.write(buf)
    sys.stdout.buffer.write(buf.getvalue())


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover
    cli.main(args=list(argv) if argv is not None else None, standalone_mode=False)


if __name__ == "__main__":  # pragma: no cover
    main()
