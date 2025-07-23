"""Improved n-gram drum groove sampler.

This module implements a memory efficient n-gram model for drum loop
generation. Contexts are hashed using MurmurHash3 and frequency counts are
stored in ``numpy`` arrays.  A coarse resolution option groups intra-bar
positions into four buckets.
"""

from __future__ import annotations

import json
import logging
import pickle
import sys
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from random import Random

import pretty_midi

log = logging.getLogger(__name__)

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from utilities.loop_ingest import load_meta

logger = logging.getLogger(__name__)

try:
    from .groove_sampler import _PITCH_TO_LABEL, _iter_drum_notes, infer_resolution
except ImportError:  # fallback when executed as a script
    import os

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from utilities.groove_sampler import (
        _PITCH_TO_LABEL,
        _iter_drum_notes,
        infer_resolution,
    )

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
    rounded_end = length & 0xFFFFFFFC  # round down to 4 byte block

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
    h ^= h >> 16
    h = (h * 0x85EBCA6B) & 0xFFFFFFFF
    h ^= h >> 13
    h = (h * 0xC2B2AE35) & 0xFFFFFFFF
    h ^= h >> 16
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
    aux_key: str | None = None
    aux_counts: dict[int, dict[str, np.ndarray]] | None = None
    files_scanned: int = 0
    files_skipped: int = 0
    total_events: int = 0


def _encode_state(bar_mod: int, bin_in_bar: int, label: str) -> State:
    return bar_mod, bin_in_bar, label


def _hash_ctx(ctx: Iterable[int]) -> int:
    arr = np.fromiter(ctx, dtype=np.uint32)
    return _murmurhash3(arr.tobytes())


logger = logging.getLogger(__name__)


def convert_wav_to_midi(
    path: Path, *, fixed_bpm: float | None = None
) -> "pretty_midi.PrettyMIDI" | None:
    """Convert a WAV file to a PrettyMIDI object containing a single drum track
    (kick on MIDI pitch 36).  Returns ``None`` and logs a warning on failure.

    Parameters
    ----------
    path
        Audio file path.
    fixed_bpm
        If given, use this BPM instead of automatic tempo estimation.
    """
    try:
        import soundfile as sf
        import numpy as np
    except Exception as exc:  # pragma: no cover
        logger.warning("Audio-to-MIDI failed for %s: %s", path, exc)
        return None

    try:
        y, sr = sf.read(path, dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)

        bpm = float(fixed_bpm) if fixed_bpm is not None else 120.0

        threshold = 0.2
        min_gap = int(0.1 * sr)
        onset_samples = []
        last_onset = -min_gap
        for i, val in enumerate(y):
            if val > threshold and i - last_onset >= min_gap:
                onset_samples.append(i)
                last_onset = i

        onset_times = [s / sr for s in onset_samples]

        pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
        drum = pretty_midi.Instrument(program=0, is_drum=True)
        for t in onset_times:
            note = pretty_midi.Note(
                velocity=100,
                pitch=36,  # kick
                start=float(t),
                end=float(t) + 0.10,
            )
            drum.notes.append(note)
        pm.instruments.append(drum)
        return pm
    except Exception as exc:  # pragma: no cover
        logger.warning("Audio-to-MIDI failed for %s: %s", path, exc)
        return None


def midi_to_events(pm: "pretty_midi.PrettyMIDI") -> List[Tuple[float, int]]:
    """Extract ``(beat, pitch)`` tuples from a PrettyMIDI drum track."""
    _times, tempi = pm.get_tempo_changes()
    tempo = float(tempi[0]) if len(tempi) > 0 else 120.0
    sec_per_beat = 60.0 / tempo

    events: list[tuple[float, int]] = []
    for inst in pm.instruments:
        if not inst.is_drum:
            continue
        for n in inst.notes:
            beat = n.start / sec_per_beat
            events.append((beat, n.pitch))
    events.sort(key=lambda x: x[0])
    return events


def collect_files(root: Path, include_audio: bool = True) -> List[Path]:
    """Recursively collect MIDI and (optionally) WAV files under *root*."""
    exts = {".mid", ".midi"}
    if include_audio:
        exts |= {".wav", ".wave"}
    return [p for p in root.rglob("*") if p.suffix.lower() in exts]


def train(
    loop_dir: Path,
    *,
    n: int = 4,
    auto_res: bool = False,
    coarse: bool = False,
    beats_per_bar: int | None = None,
    n_jobs: int = 1,
    memmap_dir: Path | None = None,
    fixed_bpm: float | None = None,
    progress: bool = False,
    include_audio: bool = True,
    aux_key: str | None = None,
):
    """Build a hashed n‑gram model from drum loops located in *loop_dir*."""

    from concurrent.futures import ProcessPoolExecutor, as_completed

    paths = collect_files(loop_dir, include_audio)
    total_events = 0
    files_skipped = 0

    if not paths:
        raise SystemExit("No files found — training aborted")

    def _load(p: Path):
        try:
            if p.suffix.lower() in {".wav", ".wave"}:
                pm = convert_wav_to_midi(p, fixed_bpm=fixed_bpm)
                if pm is None:
                    return None
            else:
                pm = pretty_midi.PrettyMIDI(str(p))
            notes = midi_to_events(pm)
            offs = [off for off, _ in notes]
            return notes, offs
        except Exception as exc:
            logger.warning("Failed to load %s: %s", p, exc)
            return None

    if n_jobs == 1 or n_jobs == 0:
        loaded = [_load(p) for p in paths]
    else:
        loaded = Parallel(n_jobs=n_jobs)(delayed(_load)(p) for p in paths)

    aux_values: list[str | None] = []
    results = []
    for p, r in zip(paths, loaded):
        if r is None:
            files_skipped += 1
        else:
            results.append(r)
            meta = load_meta(p)
            aux_values.append(meta.get(aux_key) if aux_key else None)

    if not results:
        raise SystemExit("No events collected — training aborted")

    note_seqs = [r[0] for r in results]
    all_offsets = [off for r in results for off in r[1]]

    resolution = int(
        infer_resolution(all_offsets, beats_per_bar=beats_per_bar) if auto_res else 16
    )
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

    total_events = sum(len(s) for s in seqs)

    n_states = len(idx_to_state)
    freq: list[FreqTable] = [dict() for _ in range(n)]
    bucket_freq: dict[int, np.ndarray] = {}

    aux_counts: dict[int, dict[str, np.ndarray]] = {}

    for seq, aux_val in zip(seqs, aux_values):
        for i, cur in enumerate(seq):
            # unigram
            arr0 = freq[0].setdefault(0, np.zeros(n_states, dtype=np.uint32))
            arr0[cur] += 1
            if aux_val is not None:
                aux_map = aux_counts.setdefault(0, {})
                arr_aux = aux_map.get(aux_val)
                if arr_aux is None:
                    arr_aux = np.zeros(n_states, dtype=np.uint32)
                    aux_map[aux_val] = arr_aux
                arr_aux[cur] += 1

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
                if aux_val is not None:
                    aux_map = aux_counts.setdefault(ctx_id, {})
                    arr_aux = aux_map.get(aux_val)
                    if arr_aux is None:
                        arr_aux = np.zeros(n_states, dtype=np.uint32)
                        aux_map[aux_val] = arr_aux
                    arr_aux[cur] += 1

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

    logger.info(
        "Scanned %d files (skipped %d) \u2192 %d events \u2192 %d states",
        len(paths),
        files_skipped,
        total_events,
        len(idx_to_state),
    )
    if total_events == 0 or len(idx_to_state) == 0:
        raise SystemExit("No events collected - check your data directory")

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
        aux_key=aux_key,
        aux_counts=aux_counts,
        files_scanned=len(paths),
        files_skipped=files_skipped,
        total_events=total_events,
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


def _filter_probs(
    probs: np.ndarray, *, top_k: int | None = None, top_p: float | None = None
) -> np.ndarray:
    """Apply top-k and nucleus filtering to an array of probabilities."""

    if top_k is not None and 0 < top_k < len(probs):
        idx = np.argpartition(probs, len(probs) - top_k)[len(probs) - top_k :]
        mask = np.zeros_like(probs, dtype=bool)
        mask[idx] = True
        probs = np.where(mask, probs, 0)
    if top_p is not None and 0 < top_p < 1:
        order = np.argsort(probs)[::-1]
        sorted_probs = probs[order]
        cumulative = np.cumsum(sorted_probs)
        cutoff = top_p * cumulative[-1]
        mask_ord = cumulative <= cutoff
        if not mask_ord.any():
            mask_ord[0] = True
        keep = order[mask_ord]
        mask = np.zeros_like(probs, dtype=bool)
        mask[keep] = True
        probs = np.where(mask, probs, 0)
    return probs


def sample_next(
    model: NGramModel,
    history: list[int],
    bucket: int,
    rng: Random,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    cond_kick: str | None = None,
    cond: dict[str, str] | None = None,
) -> int:
    """Sample next state index using hashed back-off.

    Parameters
    ----------
    top_k:
        Limit choices to the ``k`` highest-probability states.
    top_p:
        Nucleus sampling threshold applied after ``top_k``.
    """

    n = model.n
    aux_val = None
    if cond is not None and model.aux_key:
        aux_val = cond.get(model.aux_key)
    for order in range(min(len(history), n - 1), 0, -1):
        ctx_id = _hash_ctx(history[-order:])
        arr = None
        if aux_val is not None and model.aux_counts is not None:
            arr = model.aux_counts.get(ctx_id, {}).get(aux_val)
            if arr is not None:
                arr = arr.astype(float)
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
            probs = _filter_probs(probs, top_k=top_k, top_p=top_p)
            total = probs.sum()
            if total == 0:
                return rng.randrange(len(model.idx_to_state))
            probs /= total
            return _choose(probs, rng)

    arr = None
    if aux_val is not None and model.aux_counts is not None:
        arr = model.aux_counts.get(0, {}).get(aux_val)
        if arr is not None:
            arr = arr.astype(float)
    if arr is None and model.prob is not None:
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
        probs = _filter_probs(probs, top_k=top_k, top_p=top_p)
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
        probs = _filter_probs(probs, top_k=top_k, top_p=top_p)
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
    top_k: int | None = None,
    top_p: float | None = None,
    temperature_end: float | None = None,
    seed: int | None = None,
    cond_velocity: str | None = None,
    cond_kick: str | None = None,
    cond: dict[str, str] | None = None,
) -> list[dict[str, float | str]]:
    """Generate a sequence of drum events.

    Parameters
    ----------
    temperature:
        Starting sampling temperature.
    top_k:
        If set, restrict sampling to the ``k`` most probable states.
    top_p:
        If set, restrict choices to the smallest set of states whose
        cumulative probability mass exceeds this value.
    temperature_end:
        Optional final temperature for linear scheduling.
    cond:
        Dictionary of auxiliary conditions such as style or feel.
    """

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

        if temperature_end is not None:
            prog = next_bin / end_bin
            temp = temperature + (temperature_end - temperature) * prog
        else:
            temp = temperature

        idx = sample_next(
            model,
            history,
            bucket,
            rng,
            temperature=temp,
            top_k=top_k,
            top_p=top_p,
            cond_kick=cond_kick,
            cond=cond,
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
                e["instrument"] == "hh_pedal"
                and 0 < e["offset"] - offset <= (4 / model.resolution)
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
        "aux_key": model.aux_key,
        "aux_counts": model.aux_counts,
        "files_scanned": model.files_scanned,
        "files_skipped": model.files_skipped,
        "total_events": model.total_events,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(data, fh)


def load(path: Path) -> NGramModel:
    with path.open("rb") as fh:
        data = pickle.load(fh)
    model = NGramModel(
        **{
            k: data.get(k)
            for k in [
                "n",
                "resolution",
                "resolution_coarse",
                "state_to_idx",
                "idx_to_state",
                "freq",
                "bucket_freq",
                "ctx_maps",
                "prob_paths",
            ]
        },
        prob=None,
        aux_key=data.get("aux_key"),
        aux_counts=data.get("aux_counts"),
        files_scanned=data.get("files_scanned", 0),
        files_skipped=data.get("files_skipped", 0),
        total_events=data.get("total_events", 0),
    )
    if model.prob_paths is not None:
        from .memmap_utils import load_memmap

        prob_arrays = []
        for order, p in enumerate(model.prob_paths):
            shape = (len(model.ctx_maps[order]), len(model.idx_to_state))
            prob_arrays.append(load_memmap(Path(p), shape=shape))
        model.prob = prob_arrays
    return model


def _cmd_train(args: list[str], *, quiet: bool = False, no_tqdm: bool = False) -> None:
    import argparse

    parser = argparse.ArgumentParser(prog="groove_sampler_v2 train")
    parser.add_argument("loop_dir", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=Path("model.pkl"))
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--auto-res", action="store_true")
    parser.add_argument("--coarse", action="store_true")
    parser.add_argument(
        "--beats-per-bar",
        type=int,
        help="override bar length when inferring resolution",
    )
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--memmap-dir", type=Path)
    parser.add_argument("--fixed-bpm", type=float)
    parser.add_argument("--aux-key", type=str, default="style")
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="ignore .wav/.wave files during training",
    )
    parser.add_argument(
        "--print-model",
        action="store_true",
        help="print model parameters after training",
    )
    ns = parser.parse_args(args)

    t0 = time.perf_counter()
    model = train(
        ns.loop_dir,
        n=ns.n,
        auto_res=ns.auto_res,
        coarse=ns.coarse,
        beats_per_bar=ns.beats_per_bar,
        n_jobs=ns.jobs,
        memmap_dir=ns.memmap_dir,
        fixed_bpm=ns.fixed_bpm,
        progress=not quiet and not no_tqdm,
        include_audio=not ns.no_audio,
        aux_key=ns.aux_key,
    )
    elapsed = time.perf_counter() - t0
    save(model, ns.output)
    print(f"model saved to {ns.output} ({elapsed:.2f}s)")
    if ns.print_model:
        try:
            print(
                json.dumps(
                    model, default=lambda o: getattr(o, "__dict__", str(o)), indent=2
                )
            )
        except Exception:
            print(model)


def _cmd_sample(args: list[str]) -> None:
    import argparse

    parser = argparse.ArgumentParser(prog="groove_sampler_v2 sample")
    parser.add_argument("model", type=Path)
    parser.add_argument("-l", "--length", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cond-velocity", choices=["soft", "hard"], default=None)
    parser.add_argument(
        "--cond-kick", choices=["four_on_floor", "sparse"], default=None
    )
    parser.add_argument("--cond-style", type=str)
    parser.add_argument("--cond-feel", type=str)
    parser.add_argument("--temperature-end", type=float)
    ns = parser.parse_args(args)

    model = load(ns.model)
    events = generate_events(
        model,
        bars=ns.length,
        temperature=ns.temperature,
        top_k=ns.top_k,
        top_p=ns.top_p,
        temperature_end=ns.temperature_end,
        seed=ns.seed,
        cond_velocity=ns.cond_velocity,
        cond_kick=ns.cond_kick,
        cond={
            k: v for k, v in {"style": ns.cond_style, "feel": ns.cond_feel}.items() if v
        },
    )
    json.dump(events, fp=sys.stdout)


def _cmd_stats(args: list[str]) -> None:
    import argparse

    parser = argparse.ArgumentParser(prog="groove_sampler_v2 stats")
    parser.add_argument("model", type=Path)
    ns = parser.parse_args(args)
    model = load(ns.model)
    print(f"n={model.n} resolution={model.resolution}")
    print(
        f"Scanned {model.files_scanned} files (skipped {model.files_skipped}) → {model.total_events} events → {len(model.idx_to_state)} states"
    )


def main(argv: list[str] | None = None) -> None:
    import sys as _sys
    import argparse

    argv = list(argv or _sys.argv[1:])
    parser = argparse.ArgumentParser(prog="groove_sampler_v2", add_help=False)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-q", "--quiet", action="store_true")
    parser.add_argument("--no-tqdm", action="store_true")
    parser.add_argument("--no-audio", action="store_true")
    ns, rest = parser.parse_known_args(argv)
    if ns.verbose and ns.quiet:
        parser.error("--verbose and --quiet cannot be used together")
    level = (
        logging.INFO if ns.verbose else logging.ERROR if ns.quiet else logging.WARNING
    )
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")

    if not rest:
        raise SystemExit("Usage: groove_sampler_v2 <command> ...")

    cmd = rest.pop(0)
    if cmd == "train":
        _cmd_train(
            rest + (["--no-audio"] if ns.no_audio else []),
            quiet=ns.quiet,
            no_tqdm=ns.no_tqdm,
        )
    elif cmd == "sample":
        _cmd_sample(rest)
    elif cmd == "stats":
        _cmd_stats(rest)
    else:
        raise SystemExit(f"Unknown command: {cmd}")


if __name__ == "__main__":  # pragma: no cover
    import sys

    main(sys.argv[1:])
