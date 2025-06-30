"""Portable n-gram groove sampler."""

from __future__ import annotations

import hashlib
import io
import json
import logging
import math
import pickle
import re
import shutil
import sys
import tempfile
import time
import warnings
from collections import Counter, OrderedDict, defaultdict
from collections.abc import Sequence
from pathlib import Path
from random import Random
from typing import Any, TypedDict

import click
import numpy as np
import pretty_midi
from typing_extensions import Required

from utilities import cli_playback
from utilities.loop_ingest import scan_loops
from utilities.types import AuxTuple

from .drum_map_registry import GM_DRUM_MAP

PPQ = 480
RESOLUTION = 16
VERSION = 1
ALPHA = 0.1

logger = logging.getLogger(__name__)


# mapping from MIDI pitch to drum label
_PITCH_TO_LABEL: dict[int, str] = {val[1]: k for k, val in GM_DRUM_MAP.items()}

State = tuple[int, str]
HashKey = tuple[State | int, ...]


class Event(TypedDict, total=False):
    """Drum event definition."""

    instrument: Required[str]
    offset: Required[float]
    duration: Required[float]
    velocity: Required[int]


def make_event(
    *,
    instrument: str,
    offset: float,
    duration: float = 0.25,
    velocity: int = 100,
    **extra: Any,
) -> Event:
    """Return an ``Event`` with defaults applied."""

    ev: Event = {
        "instrument": instrument,
        "offset": float(offset),
        "duration": float(duration),
        "velocity": int(velocity),
    }
    ev.update(extra)
    return ev


def init_history_from_events(events: Sequence[Event], order: int = 3) -> list[State]:
    """Return ``order-1`` recent ``(step,label)`` tuples from ``events``.

    Parameters
    ----------
    events : Sequence[Event]
        Input events sorted or unsorted by offset.
    order : int, optional
        N-gram order. ``order-1`` items will be returned.

    Returns
    -------
    list[State]
        Truncated history suitable for the next bar.
    """

    states: list[State] = []
    for ev in sorted(events, key=lambda e: e["offset"]):
        step = int(round(ev["offset"] * (RESOLUTION / 4)))
        if 0 <= step < RESOLUTION:
            states.append((step, ev["instrument"]))
    if order - 1 <= 0:
        return []
    return states[-(order - 1) :]



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
    aux_cache: dict[int, AuxTuple]
    use_sha1: bool
    num_tokens: int
    train_perplexity: float
    train_seconds: float




DEFAULT_AUX = {"section": "verse", "heat_bin": 0, "intensity": "mid"}

_AUX_HASH_CACHE: dict[int, AuxTuple] = {}
_AUX_TUPLE_MAP: dict[AuxTuple, int] = {}
_AUX_USE_SHA1 = False
_AUX_HASH_BITS = 64

CacheKey = tuple[HashKey, int | None]
_dist_cache: OrderedDict[CacheKey, list[tuple[State, float]]] = OrderedDict()


class _CacheList:
    __slots__ = ("val", "__weakref__")

    def __init__(self, val: list[tuple[State, float]]) -> None:
        self.val = val


_lin_prob: OrderedDict[CacheKey, _CacheList] = OrderedDict()
MAX_CACHE = 2048


def _hash_bytes(data: bytes, sha1: bool, bits: int = 64) -> int:
    """Return an integer hash of ``data`` using BLAKE2 or SHA-1."""

    size = bits // 8
    if sha1:
        return int.from_bytes(hashlib.sha1(data).digest()[:size], "big")
    return int.from_bytes(hashlib.blake2s(data, digest_size=size).digest(), "big")


def _hash_aux(aux: AuxTuple) -> int:
    """Return deterministic hash for an auxiliary tuple."""

    global _AUX_USE_SHA1, _AUX_HASH_BITS
    if aux in _AUX_TUPLE_MAP:
        return _AUX_TUPLE_MAP[aux]
    data = "|".join(aux).encode("utf-8")
    val = _hash_bytes(data, _AUX_USE_SHA1, _AUX_HASH_BITS)
    prev = _AUX_HASH_CACHE.get(val)
    if prev is not None and prev != aux:
        if not _AUX_USE_SHA1:
            warnings.warn(
                "aux hash collision detected; switching to SHA-1",
                RuntimeWarning,
            )
            _AUX_USE_SHA1 = True
            val = _hash_bytes(data, True, _AUX_HASH_BITS)
            prev = _AUX_HASH_CACHE.get(val)
        if prev is not None and prev != aux:
            if _AUX_HASH_BITS == 64:
                warnings.warn(
                    "SHA-1 collision; extending hash to 96 bits",
                    RuntimeWarning,
                )
                _AUX_HASH_BITS = 96
                val = _hash_bytes(data, True, _AUX_HASH_BITS)
                prev = _AUX_HASH_CACHE.get(val)
            if prev is not None and prev != aux:
                raise RuntimeError(f"Hash collision for {aux} vs {prev}")
    _AUX_HASH_CACHE[val] = aux
    _AUX_TUPLE_MAP[aux] = val
    return val


def _load_events(loop_dir: Path, exts: Sequence[str], progress: bool = False) -> tuple[
    list[tuple[list[State], str]],
    dict[str, float],
    dict[str, Counter[int]],
    dict[str, Counter[int]],
]:
    """Return token sequences and velocity statistics.

    Args:
        loop_dir: Folder containing loops.
        exts: Extensions to load.
        progress: Show a progress bar when more than 100 files are scanned.
    """

    entries = scan_loops(loop_dir, exts=exts, progress=progress)
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


def load_aux_map(aux_path: Path) -> dict[str, dict[str, Any]]:
    """Load auxiliary metadata from JSON or YAML."""

    if aux_path.suffix.lower() in {".yaml", ".yml"}:
        try:
            from ruamel.yaml import YAML  # type: ignore
            from ruamel.yaml.constructor import DuplicateKeyError  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise click.BadParameter(
                "Install ruamel.yaml to use YAML aux files"
            ) from exc
        yaml = YAML(typ="safe")
        yaml.allow_duplicate_keys = False
        with aux_path.open("r", encoding="utf-8") as fh:
            try:
                data = yaml.load(fh) or {}
            except DuplicateKeyError as exc:  # pragma: no cover - dupes
                raise ValueError(str(exc)) from exc
            except Exception as exc:  # pragma: no cover - invalid yaml
                raise ValueError(str(exc)) from exc
    else:
        with aux_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    assert isinstance(data, dict)
    return data


def validate_aux_map(
    aux_map: dict[str, dict[str, Any]],
    *,
    states: set[str] | None = None,
    filenames: Sequence[str] | None = None,
) -> None:
    """Validate an auxiliary metadata mapping.

    Args:
        aux_map: Mapping from filename to auxiliary metadata.
        states: Required attribute keys. Defaults to ``{"section", "heat_bin", "intensity"}``.
        filenames: Filenames that must exist in ``aux_map``.

    Raises:
        ValueError: If a required key is missing, a value is malformed or
        any filename from ``filenames`` is absent. Extra keys trigger a warning.
    """

    req = states or {"section", "heat_bin", "intensity"}
    for name, meta in aux_map.items():
        missing = [k for k in req if k not in meta]
        if missing:
            raise ValueError(
                f"aux entry '{name}' missing keys: {', '.join(missing)}"
            )
        extra = [k for k in meta if k not in req]
        if extra:
            warnings.warn(
                f"aux entry '{name}' has extra keys: {', '.join(extra)}",
                RuntimeWarning,
            )
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
    progress: bool = False,
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
        progress: Display a progress bar during training.

    Returns:
        Trained model with probability tables and statistics.
    """

    start = time.time()
    _AUX_HASH_CACHE.clear()
    _AUX_TUPLE_MAP.clear()
    global _AUX_USE_SHA1, _AUX_HASH_BITS
    _AUX_USE_SHA1 = False
    _AUX_HASH_BITS = 64
    _dist_cache.clear()
    _lin_prob.clear()

    exts = [e.strip().lower() for e in ext.split(",") if e]
    seqs, mean_vel, vel_deltas, micro_offsets = _load_events(loop_dir, exts, progress=progress)
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
    duration = time.time() - start

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
        train_seconds=duration,
    )
    return model


def save(model: Model, path: Path) -> None:
    with path.open("wb") as fh:
        pickle.dump(dict(model), fh)


def load(path: Path) -> Model:
    with path.open("rb") as fh:
        data = pickle.load(fh)
    _dist_cache.clear()
    _lin_prob.clear()
    if (
        data.get("resolution") != RESOLUTION
        or data.get("version") != VERSION
    ):
        raise RuntimeError("incompatible model version")
    # drop legacy keys from older model versions
    data.pop("aux_dims", None)
    data.setdefault("num_tokens", 0)
    data.setdefault("train_perplexity", float("inf"))
    data.setdefault("train_seconds", 0.0)
    model = Model(**data)
    _AUX_HASH_CACHE.clear()
    _AUX_HASH_CACHE.update(model.get("aux_cache", {}))
    _AUX_TUPLE_MAP.clear()
    for val, tup in _AUX_HASH_CACHE.items():
        _AUX_TUPLE_MAP[tup] = val
    global _AUX_USE_SHA1, _AUX_HASH_BITS
    _AUX_USE_SHA1 = model.get("use_sha1", False)
    big_hash = any(v > (1 << 64) - 1 for v in _AUX_HASH_CACHE)
    if big_hash and not model.get("use_sha1", False):
        warnings.warn(
            "Model uses 96-bit aux hashes; re-save to persist upgrade.",
            RuntimeWarning,
        )
        model["use_sha1"] = True
        _AUX_USE_SHA1 = True
    _AUX_HASH_BITS = 96 if big_hash else 64
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
    cache: dict[CacheKey, list[tuple[State, float]]] | None = None,
) -> State:
    """Return the next state given ``history``.

    Args:
        history: Current n-gram context.
        model: Loaded model.
        rng: Random number generator.
        cond: Optional auxiliary conditioning dictionary.
        temperature: Sampling temperature.
        top_k: If set, restrict choices to the ``k`` highest-probability states.
        cache: Optional distribution cache reused within a bar.

    Returns:
        Selected ``(step, label)`` state.
    """

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
        raise RuntimeError(
            f"No probability mass for context {ctx!r} aux {aux_full!r}"
        )
    linear_list: list[tuple[State, float]]
    key = (ctx + (aux_full,), top_k)
    if temperature <= 0:
        return max(dist, key=dist.get)
    cached = None
    if cache is not None:
        cached = cache.get(key)
    if cached is None:
        cached = _dist_cache.get(key)
        if cached is not None:
            _dist_cache.move_to_end(key)
    if cached is not None:
        linear_list = cached
    else:
        wrapper = _lin_prob.get(key)
        if wrapper is None:
            linear_list = sorted(dist.items(), key=lambda x: x[1], reverse=True)
            if top_k:
                linear_list = linear_list[:top_k]
            linear_list = [(k, float(math.exp(v))) for k, v in linear_list]
            wrapper = _CacheList(linear_list)
            _lin_prob[key] = wrapper
            if len(_lin_prob) > MAX_CACHE:
                _lin_prob.popitem(last=False)
        else:
            linear_list = wrapper.val
            _lin_prob.move_to_end(key)
        _dist_cache[key] = linear_list
        if len(_dist_cache) > MAX_CACHE:
            _dist_cache.popitem(last=False)
    if cache is not None:
        cache[key] = linear_list
    
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
    progress: bool = False,
    humanize_vel: bool = False,
    humanize_micro: bool = False,
    use_bar_cache: bool = True,
    history: list[State] | None = None,
    **kwargs: Any,
) -> list[Event]:
    """Generate a sequence of drum events.

    Args:
        model: Trained n-gram model.
        bars: Number of bars to generate.
        temperature: Sampling temperature (``0`` for deterministic).
        top_k: Restrict choices to the ``k`` most likely states.
        seed: Optional RNG seed for reproducible output.
        cond: Optional auxiliary conditioning dictionary.
        progress: Display a progress bar when ``bars`` is large.
        humanize_vel: Apply velocity deltas from the training data.
        humanize_micro: Apply micro timing offsets from the training data.
        use_bar_cache: Enable per-bar cache for benchmarking.
        history: Mutable list updated with the final history context.

    Returns:
        Sorted list of generated ``Event`` objects.
    """

    if "no_bar_cache" in kwargs:
        use_bar_cache = not bool(kwargs.pop("no_bar_cache"))
    if kwargs:
        raise TypeError(f"unexpected keys: {', '.join(kwargs)}")

    rng = Random(seed)
    events: list[Event] = []
    history_arg = history
    history = list(history) if history is not None else []

    iterable = range(bars)
    bar_obj = None
    if progress and bars >= 128:
        try:
            from tqdm import tqdm  # type: ignore

            bar_obj = tqdm(range(bars), unit="bar")
            iterable = bar_obj  # type: ignore[assignment]
        except Exception:
            pass

    for bar in iterable:
        bar_events, history = _generate_bar(
            history,
            model,
            temperature=temperature,
            top_k=top_k,
            cond=cond,
            humanize_vel=humanize_vel,
            humanize_micro=humanize_micro,
            use_bar_cache=use_bar_cache,
            rng=rng,
        )
        for ev in bar_events:
            ev["offset"] += bar * 4
        events.extend(bar_events)

    if bar_obj is not None:
        bar_obj.close()

    events.sort(key=lambda x: x["offset"])
    if history_arg is not None:
        history_arg[:] = history
    return events


def _generate_bar(
    prev_history: list[State] | None,
    model: Model,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    cond: dict[str, Any] | None = None,
    humanize_vel: bool = False,
    humanize_micro: bool = False,
    use_bar_cache: bool = True,
    rng: Random | None = None,
    bar_cache: dict[CacheKey, list[tuple[State, float]]] | None = None,
    **kwargs: Any,
) -> tuple[list[Event], list[State]]:
    """Generate one bar of events.

    Args:
        prev_history: Previous n-gram history or ``None`` to start fresh.
        model: Trained n-gram model to sample from.
        temperature: Sampling temperature. ``0`` selects the most probable state.
        top_k: Restrict choices to the ``k`` most likely states.
        cond: Optional auxiliary conditioning dictionary.
        humanize_vel: Apply velocity jitter from ``vel_deltas``.
        humanize_micro: Apply micro timing offsets from ``micro_offsets``.
        use_bar_cache: Enable per-bar distribution cache.
        rng: Optional random number generator.

    Returns:
        Tuple ``(events, history)`` where ``events`` is a list of ``Event``
        objects and ``history`` holds the updated n-gram context.
    """

    rand = rng or Random()
    if "no_bar_cache" in kwargs:
        use_bar_cache = not bool(kwargs.pop("no_bar_cache"))
    if kwargs:
        raise TypeError(f"unexpected keys: {', '.join(kwargs)}")

    history = list(prev_history) if prev_history is not None else []
    events: list[Event] = []
    if bar_cache is None:
        bar_cache = {} if use_bar_cache else None
    vel_bounds = {
        k: min(float(np.percentile(np.abs(list(c.elements())), 95)), 45)
        if list(c.elements())
        else 0.0
        for k, c in model.get("vel_deltas", {}).items()
    }
    micro_bounds = {
        k: min(float(np.percentile(np.abs(list(c.elements())), 95)), 30)
        if list(c.elements())
        else 0.0
        for k, c in model.get("micro_offsets", {}).items()
    }

    next_bin = 0
    while next_bin < RESOLUTION:
        state = _sample_next(
            history,
            model,
            rand,
            temperature=temperature,
            top_k=top_k,
            cond=cond,
            cache=bar_cache,
        )
        step, lbl = state
        if step < 0 or step >= RESOLUTION:
            warnings.warn("step out of range; dropped", RuntimeWarning)
            if __debug__:
                logger.debug("invalid step %s, dropped", step)
            continue
        if step < next_bin:
            step = next_bin
        offset_beats = step / (RESOLUTION / 4)
        micro = 0
        if humanize_micro:
            choices = list(model["micro_offsets"].get(lbl, Counter()).elements())
            if choices:
                micro = int(rand.choice(choices))
                limit = micro_bounds.get(lbl, 30)
                micro = max(-limit, min(limit, micro))
            else:
                micro = 0
        vel_mean = int(model["mean_velocity"].get(lbl, 100))
        vel = vel_mean
        if humanize_vel:
            choices = list(model["vel_deltas"].get(lbl, {}).elements())
            if choices:
                delta = int(rand.choice(choices))
                limit = vel_bounds.get(lbl, 45)
                delta = max(-limit, min(limit, delta))
                vel += delta
        vel = max(1, min(127, vel))
        ev: Event = {
            "instrument": lbl,
            "offset": offset_beats + micro / PPQ,
            "duration": 0.25,
            "velocity": vel,
        }
        events.append(ev)
        history.append((step, lbl))
        if len(history) > model["order"] - 1:
            history.pop(0)
        next_bin = step + 1

    events.sort(key=lambda x: x["offset"])
    return events, history


def generate_bar(
    history: list[State] | None = None,
    *,
    model: Model,
    temperature: float = 1.0,
    top_k: int | None = None,
    cond: dict[str, Any] | None = None,
    humanize_vel: bool = False,
    humanize_micro: bool = False,
    use_bar_cache: bool = True,
) -> list[Event]:
    """Return a single bar of events."""

    hist = list(history) if history is not None else None
    bar_cache: dict[CacheKey, list[tuple[State, float]]] | None
    bar_cache = {} if use_bar_cache else None
    events, hist_out = _generate_bar(
        hist,
        model,
        temperature=temperature,
        top_k=top_k,
        cond=cond,
        humanize_vel=humanize_vel,
        humanize_micro=humanize_micro,
        use_bar_cache=use_bar_cache,
        bar_cache=bar_cache,
    )
    if history is not None:
        history[:] = hist_out
    return events


def generate_bar_legacy(
    history: list[State] | None,
    model: Model,
    **kw: Any,
) -> tuple[list[Event], list[State]]:
    """Return events, history (legacy API)."""
    warnings.warn(
        "generate_bar_legacy is deprecated; use generate_bar",
        DeprecationWarning,
        stacklevel=2,
    )
    events = generate_bar(history, model=model, **kw)
    return events, history if history is not None else []

def events_to_midi(events: Sequence[Event]) -> pretty_midi.PrettyMIDI:
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
@click.option("--progress/--no-progress", default=True, help="Show progress bar")
def train_cmd(
    loop_dir: Path,
    ext: str,
    order: str,
    smoothing: str,
    alpha: float,
    discount: float,
    out_path: Path,
    aux_path: Path | None,
    progress: bool,
) -> None:
    """Train a groove model from loops.

    Args:
        loop_dir: Folder containing MIDI or WAV loops.
        ext: Comma separated list of file extensions to scan.
        order: N-gram order or ``"auto"`` for perplexity-based selection.
        smoothing: Probability smoothing method.
        alpha: Add-alpha constant when using additive smoothing.
        discount: Discount value for Kneser–Ney.
        out_path: Output path for the pickled model.
        aux_path: Optional JSON/YAML mapping of loop names to aux info.
        progress: Display a progress bar during training.
    """

    aux_map = None
    if aux_path and aux_path.exists():
        aux_map = load_aux_map(aux_path)
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
        progress=progress,
    )
    save(model, out_path)
    click.echo(f"saved model to {out_path}")


@cli.command()
@click.argument("model_path", type=Path)
@click.option(
    "--list-aux",
    "-L",
    "--aux-list",
    "list_aux",
    is_flag=True,
    help="List known aux tuples and exit",
)
@click.option(
    "--use-bar-cache/--no-bar-cache",
    default=True,
    help="Enable per-bar cache",
)
@click.option("-l", "--length", default=4, type=int)
@click.option("--temperature", default=1.0, type=float)
@click.option("--seed", default=42, type=int)
@click.option(
    "--cond",
    default=None,
    help="JSON aux condition, e.g. '{\"section\":\"chorus\"}'",
)
@click.option("--progress/--no-progress", default=True, help="Show progress bar")
@click.option(
    "--humanize",
    default="",
    help="Comma separated options: 'vel', 'micro'",
)
@click.option(
    "--play",
    is_flag=True,
    help="Preview MIDI via timidity/fluidsynth, afplay or wmplayer",
)
def sample_cmd(
    model_path: Path,
    length: int,
    temperature: float,
    seed: int,
    cond: str | None,
    list_aux: bool,
    use_bar_cache: bool,
    progress: bool,
    humanize: str,
    play: bool,
) -> None:
    """Generate MIDI from a trained model.

    Args:
        model_path: Path to the model ``.pkl`` file.
        length: Number of bars to generate.
        temperature: Sampling temperature.
        seed: Random seed for deterministic output.
        cond: JSON mapping of auxiliary conditions.
        list_aux: List available aux tuples instead of generating.
        use_bar_cache: Enable per-bar cache; disable for benchmarking.
        progress: Display a progress bar when ``length`` is large.
        humanize: Comma separated options ``"vel"`` and/or ``"micro"``.

    Notes:
        When ``--list-aux`` or ``length`` equals ``0`` the function prints
        available auxiliary combinations. Otherwise a MIDI file is written to
        ``stdout``.
    """

    model = load(model_path)
    combos = sorted(model.get("aux_cache", {}).values())
    cond_map = json.loads(cond) if cond else None
    if cond_map:
        for k in list(cond_map.keys()):
            if k not in {"section", "heat_bin", "intensity"}:
                warnings.warn(f"unknown condition key: {k}", RuntimeWarning)
                cond_map.pop(k)
    if list_aux or length == 0:
        if cond_map:
            combos = [
                tup
                for tup in combos
                if (
                    ("section" not in cond_map or cond_map["section"] == tup[0])
                    and (
                        "heat_bin" not in cond_map
                        or str(cond_map["heat_bin"]) == tup[1]
                    )
                    and (
                        "intensity" not in cond_map
                        or cond_map["intensity"] == tup[2]
                    )
                )
            ]
        click.echo(json.dumps(combos))
        return
    h_opts = {opt.strip() for opt in humanize.split(',') if opt.strip()}
    ev = sample(
        model,
        bars=length,
        temperature=temperature,
        seed=seed,
        cond=cond_map,
        progress=progress,
        humanize_vel="vel" in h_opts,
        humanize_micro="micro" in h_opts,
        use_bar_cache=use_bar_cache,
    )
    pm = events_to_midi(ev)
    buf = io.BytesIO()
    pm.write(buf)
    if play:
        data = buf.getvalue()
        player = cli_playback.find_player()
        if player:
            player(data)
        else:
            logger.warning("no MIDI player found; writing to stdout")
            sys.stdout.buffer.write(data)
    else:
        sys.stdout.buffer.write(buf.getvalue())


@cli.command()
@click.argument("model_path", type=Path)
@click.option("--json", "as_json", is_flag=True, help="Emit JSON summary")
@click.option("--stats", is_flag=True, help="Include perplexity and token count")
def info_cmd(model_path: Path, as_json: bool, stats: bool) -> None:
    """Show model statistics.

    Args:
        model_path: Path to the pickled model.
        as_json: Emit machine-readable JSON instead of plain text.
        stats: Include token and perplexity statistics as well as a
            per-instrument token histogram.
    """

    model = load(model_path)
    order = model["order"]
    aux_tuples = sorted(set(model.get("aux_cache", {}).values()))
    tokens = model.get("num_tokens", 0)
    ppx = model.get("train_perplexity", float("inf"))
    pkl_bytes = model_path.read_bytes()
    size_b = len(pkl_bytes)
    sha1 = hashlib.sha1(pkl_bytes).hexdigest()[:8]
    token_hist: dict[str, int] = {}
    if stats:
        counter = Counter()
        for (step, inst), cnt in model.get("freq", {}).get(0, {}).get((), Counter()).items():
            counter[inst] += cnt
        token_hist = dict(sorted(counter.items()))
    data = {
        "order": order,
        "resolution": model.get("resolution", RESOLUTION),
        "num_tokens": tokens,
        "perplexity": ppx,
        "unique_aux": len(aux_tuples),
        "aux_sample": aux_tuples[:5],
        "size_bytes": size_b,
        "sha1": sha1,
    }
    if stats:
        data["tokens_per_instrument"] = token_hist
    if as_json:
        click.echo(json.dumps(data))
    else:
        click.echo(f"order: {order}")
        click.echo(f"resolution: {data['resolution']}")
        click.echo(f"num_tokens: {tokens}")
        click.echo(f"perplexity: {ppx:.2f}")
        click.echo(
            "aux_sample: "
            + (", ".join("|".join(t) for t in aux_tuples[:5]) if aux_tuples else "n/a")
        )
        click.echo(f"unique_aux: {len(aux_tuples)}")
        click.echo(f"size_bytes: {size_b}")
        click.echo(f"sha1: {sha1}")
        if stats and token_hist:
            click.echo("tokens_per_instrument:")
            for inst, cnt in token_hist.items():
                click.echo(f"  {inst}: {cnt}")


def profile_train_sample() -> tuple[float, float]:
    """Measure training and sampling speed.

    Returns:
        Tuple ``(train_seconds, sample_seconds_per_bar)`` computed from
        a synthetic 500-loop workload and 256-bar generation.
    """

    tmp = Path(tempfile.mkdtemp())
    try:
        for i in range(500):
            pm = pretty_midi.PrettyMIDI(initial_tempo=120)
            inst = pretty_midi.Instrument(program=0, is_drum=True)
            for j in range(4):
                inst.notes.append(
                    pretty_midi.Note(
                        velocity=100,
                        pitch=36,
                        start=j * 0.25,
                        end=j * 0.25 + 0.1,
                    )
                )
            pm.instruments.append(inst)
            pm.write(str(tmp / f"{i}.mid"))
        t0 = time.time()
        model = train(tmp, order=1)
        train_time = time.time() - t0
        t0 = time.time()
        sample(model, bars=256)
        samp_time = time.time() - t0
    finally:
        shutil.rmtree(tmp)
    return train_time, samp_time / 256

__all__ = [
    "DEFAULT_AUX",
    "load_aux_map",
    "validate_aux_map",
    "auto_select_order",
    "train",
    "save",
    "load",
    "sample",
    "generate_bar_legacy",
    "events_to_midi",
    "profile_train_sample",
    "cli",
    "main",
]


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover
    cli.main(args=list(argv) if argv is not None else None, standalone_mode=False)


if __name__ == "__main__":  # pragma: no cover
    main()

