"""Improved n-gram drum groove sampler.

This module implements a memory efficient n-gram model for drum loop
generation. Contexts are hashed using MurmurHash3 and frequency counts are
stored in ``numpy`` arrays.  A coarse resolution option groups intra-bar
positions into four buckets.
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from random import Random

import numpy as np
from joblib import Parallel, delayed

from .groove_sampler import _PITCH_TO_LABEL, _iter_drum_notes, infer_resolution

State = tuple[int, int, str]
"""Model state encoded as ``(bar_mod2, bin_in_bar, drum_label)``."""

FreqTable = dict[int, np.ndarray]
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
    state_to_idx: dict[State, int]
    idx_to_state: list[State]
    freq: list[FreqTable]
    bucket_freq: dict[int, np.ndarray]
    ctx_maps: list[dict[int, int]]
    prob_paths: list[str] | None = None
    prob: list[np.ndarray] | None = None


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
    n_jobs: int = -1,
    memmap_dir: Path | None = None,
) -> NGramModel:
    """Build a hashed n-gram model from ``loop_dir``."""

    paths = list(loop_dir.glob("*.mid"))

    def _load(p: Path) -> tuple[list[tuple[float, int]], list[float]]:
        notes = _iter_drum_notes(p)
        offs = [off for off, _ in notes]
        return notes, offs

    if paths:
        if n_jobs == 1:
            results = [_load(p) for p in paths]
        else:
            results = Parallel(n_jobs=n_jobs)(delayed(_load)(p) for p in paths)
        note_seqs = [r[0] for r in results if r[0]]
        all_offsets = [off for r in results for off in r[1]]
    else:
        note_seqs = []
        all_offsets = []

    resolution = int(infer_resolution(all_offsets) if auto_res else 16)
    resolution_coarse = resolution // 4 if coarse else resolution
    step_per_beat = resolution / 4
    bar_len = resolution

    state_to_idx: dict[State, int] = {}
    idx_to_state: list[State] = []
    seqs: list[list[int]] = []

    for notes in note_seqs:
        seq: list[int] = []
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
    freq: list[FreqTable] = [dict() for _ in range(n)]
    bucket_freq: dict[int, np.ndarray] = {}

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

    ctx_maps: list[dict[int, int]] = []
    prob_arrays: list[np.ndarray] = []
    prob_paths: list[str] | None = [] if memmap_dir else None
    if memmap_dir:
        memmap_dir.mkdir(parents=True, exist_ok=True)

    for order in range(n):
        ctx_ids = list(freq[order].keys())
        ctx_map = {cid: i for i, cid in enumerate(ctx_ids)}
        ctx_maps.append(ctx_map)
        arr = np.zeros((len(ctx_ids), n_states), dtype=np.float32)
        for i, cid in enumerate(ctx_ids):
            c = freq[order][cid]
            s = c.sum()
            if s:
                arr[i] = c / s
        if memmap_dir is not None:
            path = memmap_dir / f"prob_order{order}.mmap"
            mm = np.memmap(path, dtype="float32", mode="w+", shape=arr.shape)
            mm[:] = arr
            prob_arrays.append(mm)
            assert prob_paths is not None
            prob_paths.append(str(path))
        else:
            prob_arrays.append(arr)

    return NGramModel(
        n=n,
        resolution=resolution,
        resolution_coarse=resolution_coarse,
        state_to_idx=state_to_idx,
        idx_to_state=idx_to_state,
        freq=freq,
        bucket_freq=bucket_freq,
        ctx_maps=ctx_maps,
        prob_paths=prob_paths,
        prob=prob_arrays,
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
    model: NGramModel,
    history: list[int],
    bucket: int,
    rng: Random,
    *,
    temperature: float = 1.0,
    cond_kick: str | None = None,
) -> int:
    """Sample next state index using hashed back-off."""

    n = model.n
    for order in range(min(len(history), n - 1), 0, -1):
        ctx_id = _hash_ctx(history[-order:])
        arr = None
        if model.prob is not None:
            row = model.ctx_maps[order].get(ctx_id)
            if row is not None:
                arr = np.asarray(model.prob[order][row])
        if arr is None:
            arr = model.freq[order].get(ctx_id)
            if arr is not None:
                arr = arr.astype(float)
        if arr is not None and arr.sum() > 0:
            probs = arr.astype(float)
            if cond_kick in {"four_on_floor", "sparse"}:
                for i, st in enumerate(model.idx_to_state):
                    if st[2] == "kick":
                        if cond_kick == "four_on_floor" and bucket == 0:
                            probs[i] *= 16
                        elif cond_kick == "sparse":
                            probs[i] *= 0.5
            if temperature != 1.0:
                probs = np.power(probs, 1.0 / temperature)
            total = probs.sum()
            if total == 0:
                return rng.randrange(len(model.idx_to_state))
            probs /= total
            return _choose(probs, rng)

    arr = None
    if model.prob is not None:
        arr = np.asarray(model.prob[0][model.ctx_maps[0].get(0, 0)])
    if arr is None:
        arr = model.freq[0].get(0)
        if arr is not None:
            arr = arr.astype(float)
    if arr is not None and arr.sum() > 0:
        probs = arr.astype(float)
        if cond_kick in {"four_on_floor", "sparse"}:
            for i, st in enumerate(model.idx_to_state):
                if st[2] == "kick":
                    if cond_kick == "four_on_floor" and bucket == 0:
                        probs[i] *= 16
                    elif cond_kick == "sparse":
                        probs[i] *= 0.5
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
        if cond_kick in {"four_on_floor", "sparse"}:
            for i, st in enumerate(model.idx_to_state):
                if st[2] == "kick":
                    if cond_kick == "four_on_floor" and bucket == 0:
                        probs[i] *= 16
                    elif cond_kick == "sparse":
                        probs[i] *= 0.5
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
    cond_velocity: str | None = None,
    cond_kick: str | None = None,
) -> list[dict[str, float | str]]:
    """Generate a sequence of drum events."""

    rng = Random(seed)
    res = model.resolution
    step_per_beat = res / 4
    bar_len = res
    tempo = 120
    sec_per_beat = 60 / tempo
    shift_ms = max(2.0, min(5.0, 60 / tempo / 8 * 1000))
    shift_beats = (shift_ms / 1000) / sec_per_beat
    retry_beats = (1 / 1000) / sec_per_beat
    events: list[dict[str, float | str]] = []
    history: list[int] = []

    next_bin = 0
    end_bin = bars * bar_len

    while next_bin < end_bin:
        bucket = next_bin % bar_len
        if model.resolution_coarse != res:
            bucket //= 4

        idx = sample_next(
            model,
            history,
            bucket,
            rng,
            temperature=temperature,
            cond_kick=cond_kick,
        )
        bar_mod, bin_in_bar, lbl = model.idx_to_state[idx]
        if model.resolution_coarse != res:
            bin_in_bar *= 4
        abs_bin = (next_bin // bar_len) * bar_len + bin_in_bar
        if abs_bin < next_bin:
            abs_bin = next_bin
        velocity = 1.0
        if cond_velocity == "soft":
            velocity = min(velocity, 0.8)
        elif cond_velocity == "hard":
            velocity = max(velocity, 1.2)
        offset = abs_bin / step_per_beat
        if lbl == "ghost_snare":
            offset += rng.normal(0.0, 0.003) / sec_per_beat
        skip = False
        for ev in events:
            if abs(ev["offset"] - offset) <= 1e-6:
                if lbl == "snare" and ev["instrument"] == "kick":
                    events.remove(ev)
                    break
                if lbl == "kick" and ev["instrument"] == "snare":
                    skip = True
                    break
                if lbl.endswith("hh") and ev["instrument"] in {"kick", "snare"}:
                    off = offset + shift_beats
                    for _ in range(3):
                        if not any(abs(e["offset"] - off) <= 1e-6 for e in events):
                            break
                        off += retry_beats
                    offset = off
                    break
                if lbl in {"kick", "snare"} and ev["instrument"].endswith("hh"):
                    off = ev["offset"] + shift_beats
                    for _ in range(3):
                        if not any(abs(e["offset"] - off) <= 1e-6 for e in events):
                            break
                        off += retry_beats
                    ev["offset"] = off
        if skip:
            continue
        events.append(
            {
                "instrument": lbl,
                "offset": offset,
                "duration": 0.25 / step_per_beat,
                "velocity_factor": velocity,
            }
        )
        if lbl == "ohh":
            choke_off = offset + (4 / model.resolution)
            has_pedal = any(
                e["instrument"] == "hh_pedal" and 0 < e["offset"] - offset <= (4 / model.resolution)
                for e in events
            )
            if not has_pedal and rng.random() < 0.3:
                events.append(
                    {
                        "instrument": "hh_pedal",
                        "offset": choke_off,
                        "duration": 0.25 / step_per_beat,
                        "velocity_factor": velocity,
                    }
                )
        history.append(idx)
        if len(history) > model.n - 1:
            history.pop(0)
        next_bin = abs_bin + 1

    events.sort(key=lambda e: e["offset"])
    return events


def save(model: NGramModel, path: Path) -> None:
    data = {
        "n": model.n,
        "resolution": model.resolution,
        "resolution_coarse": model.resolution_coarse,
        "state_to_idx": model.state_to_idx,
        "idx_to_state": model.idx_to_state,
        "freq": model.freq,
        "bucket_freq": model.bucket_freq,
        "ctx_maps": model.ctx_maps,
        "prob_paths": model.prob_paths,
    }
    with path.open("wb") as fh:
        pickle.dump(data, fh)


def load(path: Path) -> NGramModel:
    with path.open("rb") as fh:
        data = pickle.load(fh)
    model = NGramModel(**data, prob=None)
    if model.prob_paths is not None:
        from .memmap_utils import load_memmap

        prob_arrays = []
        for order, p in enumerate(model.prob_paths):
            shape = (len(model.ctx_maps[order]), len(model.idx_to_state))
            prob_arrays.append(load_memmap(Path(p), shape=shape))
        model.prob = prob_arrays
    return model


def _cmd_train(args: list[str]) -> None:
    import argparse

    parser = argparse.ArgumentParser(prog="groove_sampler_v2 train")
    parser.add_argument("loop_dir", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=Path("model.pkl"))
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--auto-res", action="store_true")
    parser.add_argument("--coarse", action="store_true")
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument("--memmap-dir", type=Path)
    ns = parser.parse_args(args)

    t0 = time.perf_counter()
    model = train(
        ns.loop_dir,
        n=ns.n,
        auto_res=ns.auto_res,
        coarse=ns.coarse,
        n_jobs=ns.jobs,
        memmap_dir=ns.memmap_dir,
    )
    elapsed = time.perf_counter() - t0
    save(model, ns.output)
    print(f"model saved to {ns.output} ({elapsed:.2f}s)")


def _cmd_sample(args: list[str]) -> None:
    import argparse

    parser = argparse.ArgumentParser(prog="groove_sampler_v2 sample")
    parser.add_argument("model", type=Path)
    parser.add_argument("-l", "--length", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cond-velocity", choices=["soft", "hard"], default=None)
    parser.add_argument(
        "--cond-kick", choices=["four_on_floor", "sparse"], default=None
    )
    ns = parser.parse_args(args)

    model = load(ns.model)
    events = generate_events(
        model,
        bars=ns.length,
        temperature=ns.temperature,
        seed=ns.seed,
        cond_velocity=ns.cond_velocity,
        cond_kick=ns.cond_kick,
    )
    json.dump(events, fp=sys.stdout)


def _cmd_stats(args: list[str]) -> None:
    import argparse

    parser = argparse.ArgumentParser(prog="groove_sampler_v2 stats")
    parser.add_argument("model", type=Path)
    ns = parser.parse_args(args)
    model = load(ns.model)
    print(f"n={model.n} resolution={model.resolution}")


def main(argv: list[str] | None = None) -> None:
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

