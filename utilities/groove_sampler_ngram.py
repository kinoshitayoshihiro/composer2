"""Portable n-gram groove sampler."""

from __future__ import annotations

import hashlib
import io
import json
import pickle
import sys
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from random import Random
from typing import Any, TypedDict

import click
import numpy as np
import pretty_midi

from .drum_map_registry import GM_DRUM_MAP
from groove_sampler.loop_ingest import (
    _iter_midi,
    _iter_wav,
    scan_loops as _load_events,
    LoadResult,
)

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
    # Log-probabilities for each context/state pair
    prob: dict[int, dict[HashKey, dict[State, float]]]
    mean_velocity: dict[str, float]
    vel_deltas: dict[str, Counter[int]]
    micro_offsets: dict[str, Counter[int]]




DEFAULT_AUX = {"section": "verse", "heat_bin": 0, "intensity": "mid"}

_AUX_HASH_CACHE: dict[int, tuple[str, str, str]] = {}


def _hash_aux(aux: tuple[str, str, str]) -> int:
    data = "|".join(aux).encode("utf-8")
    val = int.from_bytes(hashlib.blake2s(data, digest_size=4).digest(), "big")
    prev = _AUX_HASH_CACHE.setdefault(val, aux)
    if prev != aux:
        raise RuntimeError(f"hash collision: {prev} vs {aux}")
    return val


def _count_ngrams(
    seqs: list[list[State]], order: int
) -> dict[int, dict[tuple[State, ...], Counter[State]]]:
    """Return n-gram frequency tables for ``seqs`` up to ``order``.

    The zeroth table stores unigram counts, while higher orders index by a
    history tuple of length ``i``.
    """

    freq: dict[int, dict[tuple[State, ...], Counter[State]]] = {
        i: defaultdict(Counter) for i in range(order)
    }
    for seq in seqs:
        for s in seq:
            freq[0][()][s] += 1
        for i in range(1, order):
            for idx in range(len(seq) - i):
                ctx = tuple(seq[idx : idx + i])
                nxt = seq[idx + i]
                freq[i][ctx][nxt] += 1
    return freq


def _kneser_ney_base_unigram_prob(
    freq_0: dict[HashKey, Counter[State]],
    freq_1: dict[HashKey, Counter[State]],
    states: set[State],
    discount: float,
) -> tuple[dict[HashKey, dict[State, float]], dict[State, float]]:
    """Return unigram probabilities and base continuation distribution."""

    cont_sets: dict[State, set[HashKey]] = defaultdict(set)
    for ctx, counter in freq_1.items():
        for state in counter:
            cont_sets[state].add(ctx)
    cont_counts = {s: len(v) for s, v in cont_sets.items()}
    total_cont = sum(cont_counts.values()) or 1
    base_prob = {
        s: max(cont_counts.get(s, 0), 1e-12) / total_cont for s in states
    }

    prob0: dict[HashKey, dict[State, float]] = {}
    for ctx, counter in freq_0.items():
        total = sum(counter.values())
        if total == 0:
            raise ValueError(f"zero frequency for context {ctx}")
        lambda_w = discount * len(counter) / total
        dist: dict[State, float] = {}
        for s in states:
            p = max(counter.get(s, 0) - discount, 0) / total
            p += lambda_w * base_prob[s]
            val = float(np.log(max(p, 1e-12)))
            assert np.isfinite(val)
            dist[s] = val
        prob0[ctx] = dist
    return prob0, base_prob


def _kneser_ney_higher_orders(
    freq: dict[int, dict[HashKey, Counter[State]]],
    prob0: dict[HashKey, dict[State, float]],
    base_prob: dict[State, float],
    states: set[State],
    discount: float,
) -> dict[int, dict[HashKey, dict[State, float]]]:
    """Compute higher-order Kneser-Ney probabilities."""

    prob: dict[int, dict[HashKey, dict[State, float]]] = {0: prob0}
    max_order = max(freq) + 1
    for order_i in range(1, max_order):
        prob[order_i] = {}
        for ctx, counter in freq.get(order_i, {}).items():
            total = sum(counter.values())
            if total == 0:
                raise ValueError(f"zero frequency for context {ctx}")
            base_ctx = ctx[1:] if order_i > 1 else ()
            base_dist = prob[order_i - 1].get(base_ctx, prob[order_i - 1].get(()))
            lambda_w = discount * len(counter) / total
            dist: dict[State, float] = {}
            for s in states:
                lower_p = (
                    np.exp(base_dist.get(s, np.log(base_prob.get(s, 1e-12))))
                    if base_dist is not None
                    else base_prob.get(s, 1e-12)
                )
                p = max(counter.get(s, 0) - discount, 0) / total
                p += lambda_w * lower_p
                val = float(np.log(max(p, 1e-12)))
                assert np.isfinite(val)
                dist[s] = val
            prob[order_i][ctx] = dist
    return prob


def _freq_to_log_prob(
    freq: dict[int, dict[HashKey, Counter[State]]],
    *,
    smoothing: str,
    alpha: float,
    discount: float = 0.75,
) -> dict[int, dict[HashKey, dict[State, float]]]:
    """Convert n-gram frequencies to log-probabilities.

    ``add_alpha`` applies simple additive smoothing. ``kneser_ney`` uses
    absolute discounting with back-off to lower-order models. Both variants
    guard against zero counts and emit finite log values.

    Parameters
    ----------
    freq:
        N-gram counts indexed by history context.
    smoothing:
        Either ``"add_alpha"`` or ``"kneser_ney"``.
    alpha:
        Additive constant for add-α smoothing.
    discount:
        Discount factor for Kneser–Ney smoothing.
    """

    prob: dict[int, dict[HashKey, dict[State, float]]] = {i: {} for i in freq}

    # collect global state set
    states: set[State] = set()
    for ctx_map in freq.get(0, {}).values():
        states.update(ctx_map.keys())

    if smoothing == "kneser_ney":
        prob0, base_prob = _kneser_ney_base_unigram_prob(
            freq.get(0, {}), freq.get(1, {}), states, discount
        )
        prob.update(
            _kneser_ney_higher_orders(freq, prob0, base_prob, states, discount)
        )
    elif smoothing == "add_alpha":
        for order_i, ctx_map in freq.items():
            for ctx, counter in ctx_map.items():
                total = sum(counter.values())
                if total == 0:
                    raise ValueError(f"zero frequency for context {ctx}")
                v = len(counter)
                denom = total + alpha * v
                dist = {s: float(np.log((c + alpha) / denom)) for s, c in counter.items()}
                assert all(np.isfinite(v) for v in dist.values())
                prob[order_i][ctx] = dist
    else:
        raise ValueError(f"unknown smoothing: {smoothing}")
    return prob


def _get_log_prob(
    history: Sequence[State],
    state: State,
    prob: dict[int, dict[HashKey, dict[State, float]]],
    order: int,
) -> float:
    """Return log probability of ``state`` given ``history``."""
    for order_i in range(min(len(history), order - 1), -1, -1):
        ctx = tuple(history[-order_i:])
        dist = prob.get(order_i, {}).get(ctx)
        if dist and state in dist:
            return dist[state]
    dist = prob.get(0, {}).get((), {})
    return dist.get(state, float(np.log(1e-12)))


def _perplexity(
    prob: dict[int, dict[HashKey, dict[State, float]]],
    seqs: list[list[State]],
    order: int,
) -> float:
    """Return perplexity of ``seqs`` under ``prob``.

    The function walks each sequence once, summing log probabilities with
    back-off as necessary. The returned value is ``exp(-L/N)`` where ``L`` is the
    total log likelihood and ``N`` the number of tokens. When ``seqs`` is empty
    ``inf`` is returned.
    """
    total_log = 0.0
    count = 0
    for seq in seqs:
        history: list[State] = []
        for state in seq:
            total_log += _get_log_prob(history, state, prob, order)
            count += 1
            history.append(state)
            if len(history) > order - 1:
                history.pop(0)
    if count == 0:
        return float("inf")
    return float(np.exp(-total_log / count))


def auto_select_order(
    seqs: list[list[State]], max_order: int = 5, validation_split: float = 0.1
) -> int:
    """Return the n-gram order with minimal perplexity."""

    if not seqs:
        raise ValueError("no sequences provided")
    rng = Random(0)
    seqs_copy = seqs[:]
    rng.shuffle(seqs_copy)
    split = max(1, int(len(seqs_copy) * validation_split))
    val_seqs = seqs_copy[:split]
    train_seqs = seqs_copy[split:] or seqs_copy[:1]
    best_order = 1
    best_ppx = float("inf")
    for order in range(1, min(max_order, 5) + 1):
        freq = _count_ngrams(train_seqs, order)
        prob = _freq_to_log_prob(
            freq, smoothing="add_alpha", alpha=ALPHA, discount=0.75
        )
        ppx = _perplexity(prob, val_seqs, order)
        if ppx < best_ppx:
            best_order = order
            best_ppx = ppx
    return best_order


def train(
    loop_dir: Path,
    *,
    ext: str = "midi",
    order: int | str = "auto",
    aux_map: dict[str, dict[str, Any]] | None = None,
    smoothing: str = "add_alpha",
    alpha: float = ALPHA,
    discount: float = 0.75,
) -> Model:
    """Train an n-gram model from MIDI/WAV loops.

    If ``order`` is ``"auto"``, the order is chosen via minimal perplexity on a
    held-out validation set.
    ``discount`` controls absolute discounting for Kneser-Ney.
    """

    exts = [e.strip().lower() for e in ext.split(",") if e]
    seqs, mean_vel, vel_deltas, micro_offsets = _load_events(loop_dir, exts)
    if not seqs:
        raise ValueError("no events found")

    if order == "auto":
        n = auto_select_order([s for s, _ in seqs])
    else:
        n = int(order)
    if n < 1:
        raise ValueError("order must be >= 1")

    aux_map = aux_map or {}

    freq: dict[int, dict[tuple[State, ...], Counter[State]]] = {
        i: defaultdict(Counter) for i in range(n)
    }
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

    prob = _freq_to_log_prob(
        freq, smoothing=smoothing, alpha=alpha, discount=discount
    )

    model: Model = Model(
        version=VERSION,
        resolution=RESOLUTION,
        order=n,
        freq=freq,
        prob=prob,
        mean_velocity=mean_vel,
        vel_deltas=vel_deltas,
        micro_offsets=micro_offsets,
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
    # drop legacy keys from older model versions
    data.pop("aux_dims", None)
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

    def _aux_hash(with_intensity: bool = True) -> int:
        section = (
            str(cond.get("section", DEFAULT_AUX["section"]))
            if cond
            else DEFAULT_AUX["section"]
        )
        heat_bin = (
            str(cond.get("heat_bin", DEFAULT_AUX["heat_bin"]))
            if cond
            else str(DEFAULT_AUX["heat_bin"])
        )
        intensity = (
            str(cond.get("intensity", DEFAULT_AUX["intensity"]))
            if cond
            else DEFAULT_AUX["intensity"]
        )
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
    linear = {k: float(np.exp(v)) for k, v in dist.items()}
    if top_k is not None:
        linear = dict(sorted(linear.items(), key=lambda x: x[1], reverse=True)[:top_k])
    if temperature != 1.0:
        linear = {k: v ** (1.0 / temperature) for k, v in linear.items()}
        total = sum(linear.values())
        linear = {k: v / total for k, v in linear.items()}
    return _choose(linear, rng)


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
@click.option(
    "--order",
    default="auto",
    help="n-gram order or 'auto' for perplexity-based selection",
)
@click.option(
    "--smoothing",
    default="add_alpha",
    type=click.Choice(["add_alpha", "kneser_ney"]),
    show_default=True,
    help="Smoothing method (use 'kneser_ney' for small or sparse datasets)",
)
@click.option("--alpha", default=ALPHA, type=float, show_default=True)
@click.option("--discount", default=0.75, type=float, show_default=True)
@click.option("--out", "out_path", type=Path, default=Path("model.pkl"))
@click.option(
    "--aux",
    "aux_path",
    type=Path,
    default=None,
    help="JSON map of loop names to aux data, e.g. '{\"foo.mid\": {\"section\": \"chorus\"}}'",
)
def train_cmd(
    loop_dir: Path,
    ext: str,
    order: str,
    smoothing: str,
    alpha: float,
    discount: float,
    out_path: Path,
    aux_path: Path | None,
) -> None:
    """Train a model from loops."""

    aux_map = None
    if aux_path and aux_path.exists():
        with aux_path.open("r", encoding="utf-8") as fh:
            aux_map = json.load(fh)
    model = train(
        loop_dir,
        ext=ext,
        order=order,
        aux_map=aux_map,
        smoothing=smoothing,
        alpha=alpha,
        discount=discount,
    )
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
def sample_cmd(
    model_path: Path,
    length: int,
    temperature: float,
    seed: int,
    cond: str | None,
) -> None:
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
