"""Improved n-gram drum groove sampler.

This module implements a memory efficient n-gram model for drum loop
generation. Contexts are hashed using MurmurHash3 and frequency counts are
stored in ``numpy`` arrays.  A coarse resolution option groups intra-bar
positions into four buckets.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Dict, Iterable, List, Tuple

import json
import pickle
import time
import sys

import numpy as np

from .groove_sampler import _iter_drum_notes, infer_resolution, _PITCH_TO_LABEL


State = Tuple[int, int, str]
"""Model state encoded as ``(bar_mod2, bin_in_bar, drum_label)``."""

FreqTable = Dict[int, np.ndarray]
"""Mapping from hashed context id to next-state count array."""


def _murmurhash3(data: bytes, seed: int = 0) -> int:
    """Return unsigned 32-bit MurmurHash3 of *data*."""

    c1 = 0xCC9E2D51
    c2 = 0x1B873593
    r1 = 15
    r2 = 13
    m = 5
    n = 0xE6546B64

    length = len(data)
    h = seed & 0xFFFFFFFF
    rounded_end = (length & 0xFFFFFFFC)  # round down to 4 byte block

    for i in range(0, rounded_end, 4):
        k = int.from_bytes(data[i : i + 4], "little")
        k = (k * c1) & 0xFFFFFFFF
        k = ((k << r1) | (k >> (32 - r1))) & 0xFFFFFFFF
        k = (k * c2) & 0xFFFFFFFF

        h ^= k
        h = ((h << r2) | (h >> (32 - r2))) & 0xFFFFFFFF
        h = (h * m + n) & 0xFFFFFFFF

    k1 = 0
    tail = data[rounded_end:]
    if len(tail) == 3:
        k1 ^= tail[2] << 16
    if len(tail) >= 2:
        k1 ^= tail[1] << 8
    if len(tail) >= 1:
        k1 ^= tail[0]
        k1 = (k1 * c1) & 0xFFFFFFFF
        k1 = ((k1 << r1) | (k1 >> (32 - r1))) & 0xFFFFFFFF
        k1 = (k1 * c2) & 0xFFFFFFFF
        h ^= k1

    h ^= length
    h ^= (h >> 16)
    h = (h * 0x85EBCA6B) & 0xFFFFFFFF
    h ^= (h >> 13)
    h = (h * 0xC2B2AE35) & 0xFFFFFFFF
    h ^= (h >> 16)
    return h & 0xFFFFFFFF


@dataclass
class NGramModel:
    """Container for the hashed n-gram model."""

    n: int
    resolution: int
    resolution_coarse: int
    state_to_idx: Dict[State, int]
    idx_to_state: List[State]
    freq: List[FreqTable]
    bucket_freq: Dict[int, np.ndarray]


def _encode_state(bar_mod: int, bin_in_bar: int, label: str) -> State:
    return bar_mod, bin_in_bar, label


def _hash_ctx(ctx: Iterable[int]) -> int:
    arr = np.fromiter(ctx, dtype=np.uint32)
    return _murmurhash3(arr.tobytes())


def train(
    loop_dir: Path,
    *,
    n: int = 4,
    auto_res: bool = False,
    coarse: bool = False,
) -> NGramModel:
    """Build a hashed n-gram model from ``loop_dir``."""

    note_seqs: List[List[Tuple[float, int]]] = []
    all_offsets: List[float] = []
    for path in loop_dir.glob("*.mid"):
        notes = _iter_drum_notes(path)
        if notes:
            note_seqs.append(notes)
            all_offsets.extend(off for off, _ in notes)

    resolution = int(infer_resolution(all_offsets) if auto_res else 16)
    resolution_coarse = resolution // 4 if coarse else resolution
    step_per_beat = resolution / 4
    bar_len = resolution

    state_to_idx: Dict[State, int] = {}
    idx_to_state: List[State] = []
    seqs: List[List[int]] = []

    for notes in note_seqs:
        seq: List[int] = []
        for off, pitch in notes:
            label = _PITCH_TO_LABEL.get(pitch, str(pitch))
            bin_idx = int(round(off * step_per_beat))
            bar_mod = (bin_idx // bar_len) % 2
            bin_in_bar = bin_idx % bar_len
            if coarse:
                bin_in_bar //= 4
            st = _encode_state(bar_mod, bin_in_bar, label)
            if st not in state_to_idx:
                state_to_idx[st] = len(idx_to_state)
                idx_to_state.append(st)
            seq.append(state_to_idx[st])
        seqs.append(seq)

    n_states = len(idx_to_state)
    freq: List[FreqTable] = [dict() for _ in range(n)]
    bucket_freq: Dict[int, np.ndarray] = {}

    for seq in seqs:
        for i, cur in enumerate(seq):
            # unigram
            arr0 = freq[0].setdefault(0, np.zeros(n_states, dtype=np.uint32))
            arr0[cur] += 1

            # higher orders
            for order in range(1, n):
                if i - order < 0:
                    break
                ctx = seq[i - order : i]
                ctx_id = _hash_ctx(ctx)
                arr = freq[order].get(ctx_id)
                if arr is None:
                    arr = np.zeros(n_states, dtype=np.uint32)
                    freq[order][ctx_id] = arr
                arr[cur] += 1

            bucket = idx_to_state[cur][1]
            b_arr = bucket_freq.get(bucket)
            if b_arr is None:
                b_arr = np.zeros(n_states, dtype=np.uint32)
                bucket_freq[bucket] = b_arr
            b_arr[cur] += 1

    return NGramModel(
        n=n,
        resolution=resolution,
        resolution_coarse=resolution_coarse,
        state_to_idx=state_to_idx,
        idx_to_state=idx_to_state,
        freq=freq,
        bucket_freq=bucket_freq,
    )


def _choose(probs: np.ndarray, rng: Random) -> int:
    total = probs.sum()
    if total == 0:
        probs = np.ones_like(probs)
        total = probs.sum()
    r = rng.random() * total
    acc = 0.0
    for i, p in enumerate(probs):
        acc += p
        if r <= acc:
            return i
    return len(probs) - 1


def sample_next(
    model: NGramModel, history: List[int], bucket: int, rng: Random, *, temperature: float = 1.0
) -> int:
    """Sample next state index using hashed back-off."""

    n = model.n
    for order in range(min(len(history), n - 1), 0, -1):
        ctx_id = _hash_ctx(history[-order:])
        arr = model.freq[order].get(ctx_id)
        if arr is not None and arr.sum() > 0:
            probs = arr.astype(float)
            if temperature != 1.0:
                probs = np.power(probs, 1.0 / temperature)
            total = probs.sum()
            if total == 0:
                return rng.randrange(len(model.idx_to_state))
            probs /= total
            return _choose(probs, rng)

    arr = model.freq[0].get(0)
    if arr is not None and arr.sum() > 0:
        probs = arr.astype(float)
        if temperature != 1.0:
            probs = np.power(probs, 1.0 / temperature)
        total = probs.sum()
        if total == 0:
            return rng.randrange(len(model.idx_to_state))
        probs /= total
        return _choose(probs, rng)

    b_arr = model.bucket_freq.get(bucket)
    if b_arr is not None and b_arr.sum() > 0:
        probs = b_arr.astype(float)
        if temperature != 1.0:
            probs = np.power(probs, 1.0 / temperature)
        total = probs.sum()
        if total == 0:
            return rng.randrange(len(model.idx_to_state))
        probs /= total
        return _choose(probs, rng)

    return rng.randrange(len(model.idx_to_state))


def generate_events(
    model: NGramModel,
    *,
    bars: int = 4,
    temperature: float = 1.0,
    seed: int | None = None,
) -> List[Dict[str, float | str]]:
    """Generate a sequence of drum events."""

    rng = Random(seed)
    res = model.resolution
    step_per_beat = res / 4
    bar_len = res
    events: List[Dict[str, float | str]] = []
    history: List[int] = []

    next_bin = 0
    end_bin = bars * bar_len

    while next_bin < end_bin:
        bucket = next_bin % bar_len
        if model.resolution_coarse != res:
            bucket //= 4

        idx = sample_next(model, history, bucket, rng, temperature=temperature)
        bar_mod, bin_in_bar, lbl = model.idx_to_state[idx]
        if model.resolution_coarse != res:
            bin_in_bar *= 4
        abs_bin = (next_bin // bar_len) * bar_len + bin_in_bar
        if abs_bin < next_bin:
            abs_bin = next_bin
        velocity = 1.0
        events.append({
            "instrument": lbl,
            "offset": abs_bin / step_per_beat,
            "duration": 0.25 / step_per_beat,
            "velocity_factor": velocity,
        })
        history.append(idx)
        if len(history) > model.n - 1:
            history.pop(0)
        next_bin = abs_bin + 1

    events.sort(key=lambda e: e["offset"])
    return events


def save(model: NGramModel, path: Path) -> None:
    with path.open("wb") as fh:
        pickle.dump(model, fh)


def load(path: Path) -> NGramModel:
    with path.open("rb") as fh:
        return pickle.load(fh)


def _cmd_train(args: List[str]) -> None:
    import argparse

    parser = argparse.ArgumentParser(prog="groove_sampler_v2 train")
    parser.add_argument("loop_dir", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=Path("model.pkl"))
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--auto-res", action="store_true")
    parser.add_argument("--coarse", action="store_true")
    ns = parser.parse_args(args)

    t0 = time.perf_counter()
    model = train(ns.loop_dir, n=ns.n, auto_res=ns.auto_res, coarse=ns.coarse)
    elapsed = time.perf_counter() - t0
    save(model, ns.output)
    print(f"model saved to {ns.output} ({elapsed:.2f}s)")


def _cmd_sample(args: List[str]) -> None:
    import argparse

    parser = argparse.ArgumentParser(prog="groove_sampler_v2 sample")
    parser.add_argument("model", type=Path)
    parser.add_argument("-l", "--length", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    ns = parser.parse_args(args)

    model = load(ns.model)
    events = generate_events(model, bars=ns.length, temperature=ns.temperature, seed=ns.seed)
    json.dump(events, fp=sys.stdout)


def _cmd_stats(args: List[str]) -> None:
    import argparse

    parser = argparse.ArgumentParser(prog="groove_sampler_v2 stats")
    parser.add_argument("model", type=Path)
    ns = parser.parse_args(args)
    model = load(ns.model)
    print(f"n={model.n} resolution={model.resolution}")


def main(argv: List[str] | None = None) -> None:
    import argparse
    import sys as _sys

    argv = list(argv or _sys.argv[1:])
    if not argv:
        raise SystemExit("Usage: groove_sampler_v2 <command> ...")

    cmd = argv.pop(0)
    if cmd == "train":
        _cmd_train(argv)
    elif cmd == "sample":
        _cmd_sample(argv)
    elif cmd == "stats":
        _cmd_stats(argv)
    else:
        raise SystemExit(f"Unknown command: {cmd}")


if __name__ == "__main__":  # pragma: no cover
    import sys

    main(sys.argv[1:])

