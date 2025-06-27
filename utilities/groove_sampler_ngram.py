"""Portable n-gram groove sampler."""

from __future__ import annotations

import hashlib
import io
import json
import pickle
import re
import sys
import warnings
from collections import Counter, OrderedDict, defaultdict
from collections.abc import Sequence
from pathlib import Path
from random import Random
from typing import Any, Literal, TypedDict

import click
import numpy as np
import pretty_midi

from utilities.loop_ingest import scan_loops

from .drum_map_registry import GM_DRUM_MAP

PPQ = 480
RESOLUTION = 16
VERSION = 1
ALPHA = 0.1


# mapping from MIDI pitch to drum label
_PITCH_TO_LABEL: dict[int, str] = {val[1]: k for k, val in GM_DRUM_MAP.items()}

State = tuple[int, str]
HashKey = tuple[State | int, ...]
Intensity = Literal["low", "mid", "high"]


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
    aux_cache: dict[int, tuple[str, str, str]]
    use_sha1: bool
    num_tokens: int
    train_perplexity: float




DEFAULT_AUX = {"section": "verse", "heat_bin": 0, "intensity": "mid"}

_AUX_HASH_CACHE: dict[int, tuple[str, str, str]] = {}
_AUX_TUPLE_MAP: dict[tuple[str, str, str], int] = {}
_AUX_USE_SHA1 = False

ctx_cache: OrderedDict[HashKey, list[tuple[State, float]]] = OrderedDict()
MAX_CACHE = 4096


def _hash_bytes(data: bytes, sha1: bool) -> int:
    if sha1:
        return int.from_bytes(hashlib.sha1(data).digest()[:8], "big")
    return int.from_bytes(hashlib.blake2s(data, digest_size=8).digest(), "big")


def _hash_aux(aux: tuple[str, str, str]) -> int:
    global _AUX_USE_SHA1
    if aux in _AUX_TUPLE_MAP:
        return _AUX_TUPLE_MAP[aux]
    data = "|".join(aux).encode("utf-8")
    val = _hash_bytes(data, _AUX_USE_SHA1)
    prev = _AUX_HASH_CACHE.get(val)
    if prev is not None and prev != aux:
        if not _AUX_USE_SHA1:
            warnings.warn(
                "aux hash collision detected; switching to SHA-1",
                RuntimeWarning,
            )
            _AUX_USE_SHA1 = True
            val = _hash_bytes(data, True)
            prev = _AUX_HASH_CACHE.get(val)
        if prev is not None and prev != aux:
            warnings.warn(
                f"SHA-1 collision for {aux} vs {prev}; using fallback", RuntimeWarning
            )
            val = hash(aux) & ((1 << 64) - 1)
    _AUX_HASH_CACHE[val] = aux
    _AUX_TUPLE_MAP[aux] = val
    return val


def _load_events(loop_dir: Path, exts: Sequence[str]) -> tuple[
    list[tuple[list[State], str]],
    dict[str, float],
    dict[str, Counter[int]],
    dict[str, Counter[int]],
]:
    """Return token sequences and velocity statistics."""

    entries = scan_loops(loop_dir, exts=exts)
    events: list[tuple[list[State], str]] = []
    vel_map: dict[str, list[int]] = defaultdict(list)
    micro_map: dict[str, list[int]] = defaultdict(list)

    for entry in entries:
        seq: list[State] = []
        for step, label, vel, micro in entry["tokens"]:
            seq.append((step % RESOLUTION, label))
            vel_map[label].append(vel)
            micro_map[label].append(micro)
        if seq:
            events.append((sorted(seq, key=lambda x: x[0]), entry["file"]))

    mean_velocity = {k: sum(v) / len(v) for k, v in vel_map.items() if v}
    vel_deltas = {
        k: Counter(int(v - mean_velocity[k]) for v in vals)
        for k, vals in vel_map.items()
    }
    micro_offsets = {k: Counter(vals) for k, vals in micro_map.items()}
    return events, mean_velocity, vel_deltas, micro_offsets


def validate_aux_map(
    aux_map: dict[str, dict[str, Any]],
    *,
    states: set[str] | None = None,
    filenames: Sequence[str] | None = None,
) -> None:
    """Validate auxiliary metadata map.

    Args:
        aux_map: Mapping from filename to auxiliary attribute dict.
        states: Required keys; defaults to ``{"section", "heat_bin", "intensity"}``.
        filenames: Optional sequence of filenames that must appear in ``aux_map``.

    Raises:
        ValueError: If required keys are missing, values are invalid or any
        filename in ``filenames`` is absent.
    """

    req = states or {"section", "heat_bin", "intensity"}
    for name, meta in aux_map.items():
        missing = [k for k in req if k not in meta]
        if missing:
            raise ValueError(f"aux entry '{name}' missing keys: {', '.join(missing)}")
        section = str(meta.get("section", ""))
        if not (1 <= len(section) <= 32) or not re.fullmatch(r"[a-z0-9_-]+", section):
            raise ValueError(f"invalid section for {name}: {section!r}")
        try:
            heat = int(meta.get("heat_bin"))
        except Exception:
            raise ValueError(f"heat_bin not integer for {name}") from None
        if not 0 <= heat <= 15:
            raise ValueError(f"heat_bin out of range for {name}: {heat}")
        intensity = str(meta.get("intensity", ""))
        if intensity not in {"low", "mid", "high"}:
            raise ValueError(f"invalid intensity for {name}: {intensity!r}")
    if filenames is not None:
        missing_names = [n for n in filenames if n not in aux_map]
        if missing_names:
            raise ValueError(f"aux map missing entries for: {', '.join(missing_names)}")


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
    """Return unigram probabilities and continuation base distribution.

    Args:
        freq_0: Mapping of empty context to state counts.
        freq_1: Bigram frequencies used for continuation counts.
        states: Complete set of states in the training data.
        discount: Absolute discount value.

    Returns:
        Two dictionaries: the unigram log-probabilities indexed by the
        empty context and the base continuation probabilities for each
        state.
    """

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
    """Compute higher-order Kneser-Ney log-probabilities.

    Args:
        freq: Frequency tables for each order.
        prob0: Unigram log-probabilities.
        base_prob: Continuation probabilities from ``_kneser_ney_base_unigram_prob``.
        states: All states observed during training.
        discount: Absolute discount value.

    Returns:
        Mapping from order to context-conditioned log-probabilities.
    """

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

    Args:
        prob: Log-probability tables for each order.
        seqs: Evaluation sequences of states.
        order: Maximum history length.

    Returns:
        Perplexity value. If ``seqs`` is empty ``float('inf')`` is returned.
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
    """Return the n-gram order with minimal perplexity.

    Args:
        seqs: Training sequences used to estimate perplexity.
        max_order: Maximum order to consider.
        validation_split: Fraction of ``seqs`` to hold out for validation.

    Returns:
        The order achieving the lowest perplexity on the validation split.
    """

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
    """Train an n-gram model from loops.

    Args:
        loop_dir: Folder containing MIDI or WAV loops.
        ext: Comma separated extensions to load.
        order: N-gram order or ``"auto"`` for automatic selection.
        aux_map: Optional mapping from filename to auxiliary metadata.
        smoothing: Smoothing strategy.
        alpha: Additive constant for ``"add_alpha"`` smoothing.
        discount: Discount used by Kneser–Ney.

    Returns:
        Trained model with probability tables and statistics.
    """

    _AUX_HASH_CACHE.clear()
    _AUX_TUPLE_MAP.clear()
    global _AUX_USE_SHA1
    _AUX_USE_SHA1 = False
    ctx_cache.clear()

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
    if aux_map:
        filenames = [name for _, name in seqs]
        validate_aux_map(aux_map, filenames=filenames)

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

    num_tokens = sum(len(s) for s, _ in seqs)
    train_perplexity = _perplexity(prob, [s for s, _ in seqs], n)

    model: Model = Model(
        version=VERSION,
        resolution=RESOLUTION,
        order=n,
        freq=freq,
        prob=prob,
        mean_velocity=mean_vel,
        vel_deltas=vel_deltas,
        micro_offsets=micro_offsets,
        aux_cache=dict(_AUX_HASH_CACHE),
        use_sha1=_AUX_USE_SHA1,
        num_tokens=num_tokens,
        train_perplexity=train_perplexity,
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
    data.setdefault("num_tokens", 0)
    data.setdefault("train_perplexity", float("inf"))
    model = Model(**data)
    _AUX_HASH_CACHE.clear()
    _AUX_HASH_CACHE.update(model.get("aux_cache", {}))
    _AUX_TUPLE_MAP.clear()
    for val, tup in _AUX_HASH_CACHE.items():
        _AUX_TUPLE_MAP[tup] = val
    global _AUX_USE_SHA1
    _AUX_USE_SHA1 = model.get("use_sha1", False)
    return model


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
        val = _hash_aux((section, heat_bin, intensity))
        if val not in model["aux_cache"]:
            warnings.warn(
                f"unknown aux ({section}, {heat_bin}, {intensity}); falling back",
                RuntimeWarning,
            )
            val = _hash_aux((section, "*", "*"))
        return val

    aux_full = _aux_hash(True)
    aux_any = _aux_hash(False)
    aux_section = _hash_aux((
        str(cond.get("section", DEFAULT_AUX["section"])) if cond else DEFAULT_AUX["section"],
        "*",
        "*",
    ))
    for order in range(min(len(history), n - 1), -1, -1):
        ctx = tuple(history[-order:])
        dist = model["prob"].get(order, {}).get(ctx + (aux_full,))
        if not dist:
            dist = model["prob"].get(order, {}).get(ctx + (aux_any,))
        if not dist:
            dist = model["prob"].get(order, {}).get(ctx + (aux_section,))
        if not dist:
            dist = model["prob"].get(order, {}).get(ctx)
        if dist:
            break
    if not dist:
        dist = model["prob"][0].get((), {})
    if not dist:
        raise RuntimeError("No probability mass")
    linear_list: list[tuple[State, float]]
    key = ctx + (aux_full,)
    if top_k:
        linear_list = ctx_cache.get(key)
        if linear_list is not None:
            ctx_cache.move_to_end(key)
        if linear_list is None:
            linear_list = sorted(dist.items(), key=lambda x: x[1], reverse=True)
            ctx_cache[key] = linear_list
            if len(ctx_cache) > MAX_CACHE:
                ctx_cache.popitem(last=False)
        linear_list = linear_list[:top_k]
        linear_list = [(k, float(np.exp(v))) for k, v in linear_list]
    else:
        linear_list = [(k, float(np.exp(v))) for k, v in dist.items()]
    if temperature != 1.0:
        linear_list = [(k, v ** (1.0 / temperature)) for k, v in linear_list]
        total = sum(v for _, v in linear_list)
        linear_list = [(k, v / total) for k, v in linear_list]
    return _choose(dict(linear_list), rng)


def sample(
    model: Model,
    *,
    bars: int = 4,
    temperature: float = 1.0,
    top_k: int | None = None,
    seed: int | None = None,
    cond: dict[str, Any] | None = None,
) -> list[dict[str, float | str]]:
    """Generate a sequence of drum events.

    Args:
        model: Trained n-gram model.
        bars: Number of bars to generate.
        temperature: Sampling temperature.
        top_k: If set, restrict choices to the top ``k`` states.
        seed: Optional random seed.
        cond: Optional auxiliary conditioning dict.

    Returns:
        List of event dictionaries sorted by onset time.
    """

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
        try:
            validate_aux_map(aux_map)
        except ValueError as exc:
            raise click.BadParameter(str(exc)) from exc
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
@click.option("--list-aux", is_flag=True, help="List known aux tuples and exit")
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
    list_aux: bool,
) -> None:
    """Generate MIDI from a model.

    If ``--list-aux`` is provided or length is ``0`` the available auxiliary
    combinations are printed instead of generating events.
    """

    model = load(model_path)
    combos = sorted(model.get("aux_cache", {}).values())
    if list_aux or length == 0:
        if length == 0:
            click.echo(json.dumps(combos))
        else:
            for tup in combos:
                click.echo("|".join(tup))
        return
    cond_map = json.loads(cond) if cond else None
    ev = sample(model, bars=length, temperature=temperature, seed=seed, cond=cond_map)
    pm = events_to_midi(ev)
    buf = io.BytesIO()
    pm.write(buf)
    sys.stdout.buffer.write(buf.getvalue())


@cli.command()
@click.argument("model_path", type=Path)
@click.option("--json", "as_json", is_flag=True, help="Emit JSON summary")
@click.option("--stats", is_flag=True, help="Include perplexity and token count")
def info_cmd(model_path: Path, as_json: bool, stats: bool) -> None:
    """Print basic information about a model."""

    model = load(model_path)
    order = model["order"]
    ctxs = len(model["prob"].get(order - 1, {}))
    aux_cache = model.get("aux_cache", {})
    sections = {t[0] for t in aux_cache.values()}
    heat_bins = {t[1] for t in aux_cache.values()}
    intensities = {t[2] for t in aux_cache.values()}
    tokens = model.get("num_tokens", 0)
    ppx = model.get("train_perplexity", float("inf"))
    if as_json:
        data = {
            "order": order,
            "contexts": ctxs,
            "sections": sorted(sections),
            "heat_bins": sorted(map(int, heat_bins)),
            "intensities": sorted(intensities),
            **({"tokens": tokens, "perplexity": ppx} if stats else {}),
        }
        click.echo(json.dumps(data))
    else:
        click.echo(f"order: {order}")
        click.echo(f"contexts: {ctxs}")
        click.echo(f"sections: {', '.join(sorted(sections)) if sections else 'n/a'}")
        click.echo(
            f"heat_bins: {', '.join(sorted(map(str, heat_bins))) if heat_bins else 'n/a'}"
        )
        click.echo(
            f"intensities: {', '.join(sorted(intensities)) if intensities else 'n/a'}"
        )
        if stats:
            click.echo(f"tokens: {tokens}")
            click.echo(f"perplexity: {ppx:.2f}")


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover
    cli.main(args=list(argv) if argv is not None else None, standalone_mode=False)


if __name__ == "__main__":  # pragma: no cover
    main()
