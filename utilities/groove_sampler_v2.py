"""Improved n-gram drum groove sampler.

This module implements a memory efficient n-gram model for drum loop
generation. Contexts are hashed using 64-bit Blake2b and frequency counts are
stored in ``numpy`` arrays.  A coarse resolution option groups intra-bar
positions into four buckets.

Set ``SPARKLE_DETERMINISTIC=1`` to force deterministic RNG defaults when
sampling without an explicit seed.
"""

from __future__ import annotations

try:
    from . import pretty_midi_compat as _pmc  # ensure patching side effects
except Exception:  # pragma: no cover - best effort
    pass

import gc
import hashlib
import json
import logging
import math
import os
import pickle
import random
import re
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from math import isfinite
from pathlib import Path
from random import Random
from typing import Any

# Module-level logger for concise log control
logger = logging.getLogger(__name__)

_LEGACY_WARNED: set[str] = set()


@dataclass(frozen=True)
class _LoadWorkerConfig:
    """Configuration bundle for the parallel loader."""

    fixed_bpm: float | None
    inject_default_tempo: float
    tempo_policy: str
    fallback_bpm: float
    min_bpm: float
    max_bpm: float
    fold_halves: bool
    tag_fill_from_filename: bool
    min_bars: float
    min_notes: int
    allow_drums: bool | None = None
    allow_pitched: bool | None = None
    instrument_whitelist: set[int] | None = None


def _instrument_allowed(
    inst_program: int | None, is_drum: bool, whitelist: set[int] | None
) -> bool:
    """Return ``True`` when the instrument passes ``whitelist`` (if any)."""

    if not whitelist:
        return True
    if is_drum:
        return -1 in whitelist
    if inst_program is None:
        return True
    return inst_program in whitelist


@dataclass(frozen=True)
class _LoadResult:
    """Lightweight container returned by ``_load_worker``."""

    payload: tuple[list[tuple[float, int]], list[float], float, int, int, bool, bool] | None
    reason: str
    audio_failed: bool
    tempo: float | None
    tempo_reason: str
    tempo_source: str
    tempo_injected: bool
    tempo_error: str | None
    invalid_bpm: float | None



def _warn_ignored(name: str) -> None:
    """Emit a one-time warning when an ignored legacy argument is seen."""

    if name not in _LEGACY_WARNED:
        logger.warning("ignored legacy arg: %s", name)
        _LEGACY_WARNED.add(name)

OHH_CHOKE_DEFAULT = 0.3
_NO_TEMPO_MARKER = "__composer2_no_tempo__"

# Optional YAML (not strictly required by this module, but kept for compat)
try:  # pragma: no cover - optional dependency
    import yaml  # type: ignore
except Exception:  # pragma: no cover

    class _DummyYAML:
        @staticmethod
        def safe_load(stream):
            return {}

    yaml = _DummyYAML()  # type: ignore

# Numpy is effectively required
import numpy as np

NDArray = np.ndarray

# Process-wide RNG for deterministic sampling
_SPARKLE_DETERMINISTIC = os.getenv("SPARKLE_DETERMINISTIC") == "1"
_RNG: random.Random = random.Random(0) if _SPARKLE_DETERMINISTIC else random.Random()


def set_random_state(seed: int | None) -> None:
    """Seed the module-level RNG used for sampling."""

    global _RNG
    _RNG = random.Random(seed)


try:  # pragma: no cover - optional dependency
    import pretty_midi  # type: ignore
except Exception:  # pragma: no cover
    pretty_midi = None  # type: ignore

# pretty_midi がある環境では get_tempo_changes を (times, bpms) に矯正する
try:  # pragma: no cover - optional dependency
    from . import pretty_midi_compat as _pmc  # noqa: F401
except Exception:  # pragma: no cover
    pass
import tempfile

# Optional mido — required for training path below
try:  # pragma: no cover - optional dependency
    import mido  # type: ignore
except Exception:  # pragma: no cover
    mido = None  # type: ignore

# Optional joblib for parallel loading
try:  # pragma: no cover - optional dependency
    from joblib import Parallel, delayed  # type: ignore
except Exception:  # pragma: no cover

    class Parallel:  # type: ignore
        def __init__(self, n_jobs: int = 1, **kwargs):
            self.n_jobs = n_jobs

        def __call__(self, tasks):
            return [t() for t in tasks]

    def delayed(fn):  # type: ignore
        def _wrap(*args, **kwargs):
            return lambda: fn(*args, **kwargs)

        return _wrap


try:  # pragma: no cover - optional during lightweight testing
    from .aux_vocab import AuxVocab
except Exception:  # pragma: no cover

    class AuxVocab:  # type: ignore
        def __init__(self, *a, **k):
            self.id_to_str: list[str] = []

        def encode(self, data):
            return 0


def _rebuild_probabilities(
    model: "NGramModel",
) -> tuple[list[np.ndarray], list[dict[int, int]]]:
    """Reconstruct normalized probability tables from frequency counts."""

    vocab_size = len(model.idx_to_state)
    prob_arrays: list[np.ndarray] = []
    ctx_maps: list[dict[int, int]] = []
    for table in model.freq:
        items = sorted(table.items(), key=lambda item: item[0])
        rows = len(items)
        arr = np.zeros((rows, vocab_size), dtype=np.float32)
        ctx_map: dict[int, int] = {}
        for row, (ctx_hash, counts) in enumerate(items):
            ctx_map[ctx_hash] = row
            counts_arr = np.asarray(counts, dtype=np.float32)
            total = float(counts_arr.sum())
            if total > 0.0:
                arr[row] = counts_arr / total
        prob_arrays.append(arr)
        ctx_maps.append(ctx_map)
    return prob_arrays, ctx_maps


def _rebuild_prob_in_memory(model: "NGramModel") -> list[np.ndarray]:
    probs, ctx_maps = _rebuild_probabilities(model)
    model.ctx_maps = ctx_maps
    return probs


def _is_small_corpus(total_events: int, num_files: int, avg_events: float) -> bool:
    """Heuristic to decide when in-memory training is preferable."""

    return (total_events <= 200_000 or avg_events <= 256) and num_files <= 2_000


try:  # pragma: no cover - optional dependency
    from tqdm import tqdm  # noqa: E402
except Exception:  # pragma: no cover

    def tqdm(x, **k):
        return x


try:  # pragma: no cover - optional dependency
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import cuckoopy  # type: ignore

    _HAS_CUCKOO = True
except Exception:  # pragma: no cover
    _HAS_CUCKOO = False

from utilities.loop_ingest import load_meta  # noqa: E402

try:  # pragma: no cover - optional dependency for memmap helpers
    from .memmap_utils import load_memmap
except Exception:  # pragma: no cover - fallback using numpy.memmap

    def load_memmap(path: Path, shape: tuple[int, int]) -> np.memmap:  # type: ignore
        return np.memmap(path, mode="r", dtype=np.float32, shape=shape)

try:  # pragma: no cover - optional dependency for conditional sampling
    from .conditioning import (
        apply_feel_bias,
        apply_kick_pattern_bias,
        apply_style_bias,
        apply_velocity_bias,
    )
except Exception:  # pragma: no cover - fallback when conditioning module missing

    def apply_style_bias(probs, *args, **kwargs):  # type: ignore
        return probs

    def apply_feel_bias(probs, *args, **kwargs):  # type: ignore
        return probs

    def apply_velocity_bias(probs, *args, **kwargs):  # type: ignore
        return probs

    def apply_kick_pattern_bias(probs, *args, **kwargs):  # type: ignore
        return probs
try:  # pragma: no cover - optional dependency for GM percussion mapping
    from .gm_perc_map import label_to_number, normalize_label
except Exception:  # pragma: no cover - fallback when percussion map missing

    def normalize_label(label: str) -> str:  # type: ignore
        return str(label)

    def label_to_number(label: str) -> int:  # type: ignore
        return 0
try:  # pragma: no cover - optional dependency for hashed contexts
    from .hash_utils import hash_ctx
except Exception:  # pragma: no cover - fallback hashing when unavailable
    try:  # allow absolute import when package context differs
        from utilities.hash_utils import hash_ctx  # type: ignore
    except Exception:

        def hash_ctx(  # type: ignore
            context_events: Iterable[int], aux: tuple[int, int, int] | None = None
        ) -> int:
            data = tuple(context_events)
            if aux is not None:
                data = data + tuple(aux)
            return hash(data) & 0xFFFFFFFF

try:
    from .ngram_store import BaseNGramStore, MemoryNGramStore, SQLiteNGramStore
except Exception:  # pragma: no cover - allow absolute import when relative fails
    from utilities.ngram_store import BaseNGramStore, MemoryNGramStore, SQLiteNGramStore  # type: ignore
from .ngram_store import BaseNGramStore, MemoryNGramStore, SQLiteNGramStore

try:  # pragma: no cover - optional dependency
    from utilities.pretty_midi_safe import pm_to_mido  # noqa: E402
except Exception:  # pragma: no cover

    def pm_to_mido(pm):  # type: ignore
        class _Msg:
            def __init__(self, time: int):
                self.time = time
                self.type = "note"
                self.numerator = 4
                self.denominator = 4

        class _Midi:
            ticks_per_beat = 480
            tracks = [[_Msg(480 * 4)]]

        return _Midi()


# Salt for deterministic hashing across runs
HASH_SALT = b"composer2_groove_v2"


def _ensure_tempo(pm: pretty_midi.PrettyMIDI, default_bpm: float = 120.0) -> pretty_midi.PrettyMIDI:
    """Ensure ``pm`` has an initial tempo.

    PrettyMIDI's own tempo information is consulted first.  If no tempo is
    present, the MIDI data is searched for a ``set_tempo`` meta message via
    :mod:`mido`.  When still missing, a default tempo is injected by writing to
    a temporary MIDI file and reloading it.  ``_ensure_tempo.injected`` records
    whether a tempo was inserted.
    """

    injected = False
    _times, tempi = pm.get_tempo_changes()
    if len(tempi):
        _ensure_tempo.injected = False  # type: ignore[attr-defined]
        return pm
    if mido is None:  # pragma: no cover - dependency is required for injection
        _ensure_tempo.injected = False  # type: ignore[attr-defined]
        return pm
    try:
        midi = pm_to_mido(pm)
    except Exception as e:  # pragma: no cover - failed conversion
        logger.debug("pm_to_mido failed: %s", e)
        _ensure_tempo.injected = False  # type: ignore[attr-defined]
        return pm
    for track in midi.tracks:
        for msg in track:
            if msg.type == "set_tempo":
                _ensure_tempo.injected = False  # type: ignore[attr-defined]
                return pm
    msg = mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(default_bpm), time=0)
    midi.tracks[0].insert(0, msg)
    tmp = tempfile.NamedTemporaryFile(suffix=".mid", delete=False)
    try:
        tmp.close()
        midi.save(tmp.name)
        pm = pretty_midi.PrettyMIDI(tmp.name)
        injected = True
    except Exception as e:  # pragma: no cover
        logger.debug("PrettyMIDI tempo injection failed: %s", e)
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:  # pragma: no cover - best effort cleanup
            pass
    _ensure_tempo.injected = injected  # type: ignore[attr-defined]
    return pm


try:
    from .groove_sampler import _PITCH_TO_LABEL, infer_resolution
except ImportError:  # fallback when executed as a script
    import os as _os

    sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..")))
    from utilities.groove_sampler import _PITCH_TO_LABEL, infer_resolution

State = tuple[int, int, str]
"""Model state encoded as ``(bar_mod2, bin_in_bar, drum_label)``."""

FreqTable = dict[int, NDArray]
"""Mapping from hashed context to next-state count array."""


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
    aux_vocab: AuxVocab | None = None
    version: int = 2
    file_weights: list[float] | None = None
    files_scanned: int = 0
    files_skipped: int = 0
    files_skipped_fatal: int = 0
    files_filtered: int = 0
    total_events: int = 0
    hash_buckets: int = 16_777_216
    tempo_defaults: dict[str, float | str | None] | None = None


def _encode_state(bar_mod: int, bin_in_bar: int, label: str) -> State:
    return bar_mod, bin_in_bar, label


def _extract_aux(
    path: Path,
    *,
    aux_map: dict[str, dict[str, str]] | None = None,
    aux_key: str | None = None,
    filename_pattern: str = r"__([^-]+)-([^_]+)",
) -> dict[str, str] | None:
    """Extract auxiliary conditions for *path*.

    Priority is ``aux_map`` > ``aux_key`` > filename pattern.
    """

    if aux_map is not None and path.name in aux_map:
        return {k: str(v) for k, v in aux_map[path.name].items()}
    if aux_key is not None:
        meta = load_meta(path)
        if aux_key in meta:
            return {aux_key: str(meta[aux_key])}
    matches = dict(re.findall(filename_pattern, path.stem))
    return matches or None


def _hash64(data: bytes) -> int:
    """Return a salted 64-bit hash of *data* using Blake2b."""

    return int.from_bytes(hashlib.blake2b(data, digest_size=8, key=HASH_SALT).digest(), "little")


def _hash_ctx(ctx: Iterable[int]) -> int:
    return hash_ctx(list(ctx), (0, 0, 0))


def bump_count(table: FreqTable, key: int, tok: int, vocab_size: int) -> None:
    """Increment count for ``(key, tok)`` in ``table``."""

    arr = table.get(key)
    if arr is None:
        arr = np.zeros(vocab_size, dtype=np.uint32)
        table[key] = arr
    arr[tok] += 1


class MemmapNGramStore:
    """Disk-backed n-gram store using numpy.memmap shards."""

    def __init__(
        self,
        path: Path,
        n_orders: int,
        vocab_size: int,
        hash_buckets: int,
        dtype: str = "uint32",
        mode: str = "w+",
    ) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.n_orders = n_orders
        self.vocab_size = vocab_size
        self.hash_buckets = hash_buckets
        # dtype may be promoted during training; keep as attribute for metadata
        self.dtype = np.dtype(dtype)
        self.shard_bits = 8
        self.n_shards = 1 << self.shard_bits
        self.shard_size = hash_buckets // self.n_shards
        self.maps: list[list[np.memmap]] = []
        for order in range(n_orders):
            order_maps = []
            for shard in range(self.n_shards):
                fn = self._shard_path(order, shard)
                if fn.exists():
                    mm = np.memmap(
                        fn,
                        mode=mode,
                        dtype=self.dtype,
                        shape=(self.shard_size, vocab_size),
                    )
                    logger.debug("opened existing memmap shard %s", fn)
                else:
                    tmp = tempfile.NamedTemporaryFile(dir=self.path, delete=False)
                    try:
                        tmp.close()
                        mm = np.memmap(
                            tmp.name,
                            mode="w+",
                            dtype=self.dtype,
                            shape=(self.shard_size, vocab_size),
                        )
                        mm.flush()
                        del mm
                        gc.collect()  # ensure Windows can rename
                        os.replace(tmp.name, fn)
                    finally:
                        try:
                            os.unlink(tmp.name)
                        except OSError:
                            pass
                    mm = np.memmap(
                        fn,
                        mode=mode,
                        dtype=self.dtype,
                        shape=(self.shard_size, vocab_size),
                    )
                    logger.debug("created memmap shard %s", fn)
                order_maps.append(mm)
            self.maps.append(order_maps)

    def flush(self, tables: list[FreqTable]) -> None:
        logger.debug("flushing memmap tables (%d orders)", len(tables))
        for order, table in enumerate(tables):
            for h, arr in table.items():
                bucket = h % self.hash_buckets
                shard = bucket & (self.n_shards - 1)
                idx = bucket >> self.shard_bits
                mm = self.maps[order][shard]
                max_val = np.iinfo(mm.dtype).max
                if np.any(mm[idx] > max_val - arr):
                    self._promote(order, shard)
                    mm = self.maps[order][shard]
                mm[idx] = (mm[idx] + arr).astype(mm.dtype)
        for order_maps in self.maps:
            for mm in order_maps:
                mm.flush()

    def merge(self) -> list[FreqTable]:
        result: list[FreqTable] = [dict() for _ in range(self.n_orders)]
        for order in range(self.n_orders):
            for shard in range(self.n_shards):
                mm = self.maps[order][shard]
                for idx in range(self.shard_size):
                    arr = np.array(mm[idx], dtype=mm.dtype)
                    if arr.any():
                        bucket = (idx << self.shard_bits) | shard
                        result[order][bucket] = arr
        return result

    def _shard_path(self, order: int, shard: int) -> Path:
        return self.path / f"o{order}_{shard}.npy"

    def _promote(self, order: int, shard: int) -> None:
        """Promote a shard to uint64 when uint32 would overflow."""
        old_path = self._shard_path(order, shard)
        old_mm = self.maps[order][shard]
        shape = old_mm.shape
        old_mm.flush()
        del old_mm
        gc.collect()
        tmp = tempfile.NamedTemporaryFile(dir=self.path, delete=False)
        try:
            tmp.close()
            new_mm = np.memmap(tmp.name, mode="w+", dtype=np.uint64, shape=shape)
            data = np.memmap(old_path, mode="r", dtype=np.uint32, shape=shape)
            new_mm[:] = data[:]
            new_mm.flush()
            del new_mm, data
            gc.collect()
            os.replace(tmp.name, old_path)
            self.maps[order][shard] = np.memmap(old_path, mode="r+", dtype=np.uint64, shape=shape)
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    def write_meta(self) -> None:
        dtype = (
            "u64" if any(mm.dtype == np.uint64 for order in self.maps for mm in order) else "u32"
        )
        meta = {"schema_version": 2, "dtype": dtype}
        (self.path / "meta.json").write_text(json.dumps(meta))


def _write_prob_memmaps(model: NGramModel, directory: Path) -> None:
    """Materialize normalized probability tables as memmaps on disk."""

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    vocab_size = len(model.idx_to_state)
    prob_paths: list[str] = []
    ctx_maps: list[dict[int, int]] = []
    rows_meta: list[int] = []
    meta: dict[str, object] = {
        "schema_version": 1,
        "orders": len(model.freq),
        "vocab_size": vocab_size,
        "rows_per_order": rows_meta,
        "dtype": "float32",
    }
    for order, table in enumerate(model.freq):
        items = sorted(table.items(), key=lambda item: item[0])
        ctx_map: dict[int, int] = {}
        path = directory / f"prob_order{order}.mmap"
        tmp_path = Path(f"{path}.tmp")
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        rows = len(items)
        rows_meta.append(rows)
        if rows == 0 or vocab_size == 0:
            with open(tmp_path, "wb") as fh:
                fh.truncate(0)
            os.replace(tmp_path, path)
            prob_paths.append(str(path))
            ctx_maps.append(ctx_map)
            continue
        mm = np.memmap(tmp_path, mode="w+", dtype=np.float32, shape=(rows, vocab_size))
        for row, (ctx_hash, counts) in enumerate(items):
            ctx_map[ctx_hash] = row
            counts_arr = np.asarray(counts, dtype=np.float32)
            total = float(counts_arr.sum())
            if total > 0.0:
                mm[row] = counts_arr / total
            else:
                mm[row] = 0.0
        mm.flush()
        del mm
        gc.collect()
        os.replace(tmp_path, path)
        prob_paths.append(str(path))
        ctx_maps.append(ctx_map)
    model.prob_paths = prob_paths
    model.ctx_maps = ctx_maps
    meta_path = directory / "prob_meta.json"
    tmp_meta_path = directory / "prob_meta.json.tmp"
    with tmp_meta_path.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh)
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except OSError:
            pass
        os.replace(tmp_meta_path, meta_path)

    prob_arrays: list[np.ndarray] = []
    for order, path_str in enumerate(prob_paths):
        rows = rows_meta[order] if order < len(rows_meta) else 0
        if rows == 0 or vocab_size == 0:
            prob_arrays.append(np.zeros((rows, vocab_size), dtype=np.float32))
            continue
        prob_arrays.append(
            load_memmap(Path(path_str), shape=(rows, vocab_size))
        )
    model.prob = prob_arrays


def convert_wav_to_midi(
    path: Path, *, fixed_bpm: float | None = None
) -> pretty_midi.PrettyMIDI | None:
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
        import numpy as np
        import soundfile as sf
    except Exception as exc:  # pragma: no cover
        logger.warning("Audio-to-MIDI failed for %s: %s", path, exc)
        return None

    try:
        y, sr = sf.read(path, dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)

        if fixed_bpm is None:
            try:
                import librosa  # type: ignore
            except Exception as exc:  # pragma: no cover - optional dependency
                logger.warning(
                    "librosa unavailable (%s); using default 120 BPM for %s",
                    exc,
                    path,
                )
                tempo = 120.0
            else:
                try:
                    tempo, _ = librosa.beat.beat_track(y=y, sr=sr, trim=False)
                except Exception as exc:  # pragma: no cover
                    logger.warning(
                        "Tempo estimation failed for %s: %s; using default 120 BPM.",
                        path,
                        exc,
                    )
                    tempo = 120.0
                else:
                    if not np.isfinite(tempo) or tempo < 40 or tempo > 300:
                        logger.warning(
                            "Tempo %.1f BPM for %s out of range; using default 120 BPM.",
                            tempo,
                            path,
                        )
                        tempo = 120.0
                    else:
                        logger.info("Estimated tempo %.1f BPM for %s", tempo, path.name)
        else:
            tempo = float(fixed_bpm)

        threshold = 0.5 * np.percentile(np.abs(y), 95)
        min_gap = int((60 / tempo) / 4 * sr)
        onset_samples = []
        last_onset = -min_gap
        for i, val in enumerate(y):
            if val > threshold and i - last_onset >= min_gap:
                onset_samples.append(i)
                last_onset = i

        onset_times = [s / sr for s in onset_samples]

        pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
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


def midi_to_events(
    pm: pretty_midi.PrettyMIDI,
    tempo: float,
    *,
    allow_pitched_fallback: bool = True,
) -> list[tuple[float, int]]:
    """Extract ``(beat, pitch)`` tuples prioritizing drum instruments.

    By default the function remains drum-first, only falling back to pitched
    instruments when no drum notes exist and ``allow_pitched_fallback`` is
    enabled.

    Parameters
    ----------
    pm
        Source MIDI.
    tempo
        Tempo in beats-per-minute.  If non-positive or non-finite, a warning
        is emitted and the default 120 BPM is used.
    allow_pitched_fallback
        When ``True`` (default), fall back to pitched instruments if no drum
        notes are present.  When ``False``, only drum notes are considered.
    """
    if not np.isfinite(tempo) or tempo <= 0:
        logger.warning("Non-positive tempo %.2f detected; using default 120 BPM.", tempo)
        tempo = 120.0
    sec_per_beat = 60.0 / tempo

    drum_events: list[tuple[float, int]] = []
    pitched_events: list[tuple[float, int]] = []
    for inst in pm.instruments:
        target = drum_events if inst.is_drum else pitched_events
        for n in inst.notes:
            beat = n.start / sec_per_beat
            target.append((beat, n.pitch))

    if drum_events:
        drum_events.sort(key=lambda x: x[0])
        return drum_events

    if allow_pitched_fallback:
        pitched_events.sort(key=lambda x: x[0])
        return pitched_events

    return []


def _safe_read_bpm(
    pm: pretty_midi.PrettyMIDI,
    *,
    default_bpm: float,
    fold_halves: bool,
) -> float:
    """Return a reasonable tempo for ``pm``.

    PrettyMIDI is queried first; if it fails to provide a finite positive
    tempo, the underlying :mod:`mido` object is searched for the first
    ``set_tempo`` meta message.  If all methods fail, ``default_bpm`` is
    returned.  When ``fold_halves`` is true, tempos close to a double or
    half-time interpretation are folded into common buckets (60–180 BPM).
    ``_safe_read_bpm.last_source`` records the provenance of the returned
    tempo and can be inspected by callers for logging.
    """

    bpm: float | None = None
    source = "default"


    times, bpms = pm.get_tempo_changes()
    times_seq = times.tolist() if hasattr(times, "tolist") else list(times)
    if len(bpms) and math.isfinite(bpms[0]) and bpms[0] > 0:
        bpm = float(bpms[0])
        source = "pretty_midi"
        if len(bpms) == 1:
            t0 = float(times_seq[0]) if times_seq else 0.0
            if abs(t0) <= 1e-6 and abs(bpm - float(default_bpm)) <= 1e-6:
                bpm = float(default_bpm)
                source = "default"
    elif mido is not None:  # pragma: no branch - optional dependency
        try:
            midi = pm_to_mido(pm)
            for track in midi.tracks:
                for msg in track:
                    if msg.type == "set_tempo":
                        bpm = mido.tempo2bpm(msg.tempo)
                        source = "mido"
                        break
                if bpm is not None:
                    break
        except Exception:  # pragma: no cover - failed conversion
            pass

    if bpm is None or not math.isfinite(bpm) or bpm <= 0:
        bpm = float(default_bpm)
        source = "default"

    if getattr(pm, "_composer2_injected_tempo", False):
        source = "default"

    if fold_halves:
        base_bpm = float(bpm)
        default = float(default_bpm) if math.isfinite(default_bpm) and default_bpm > 0 else 120.0
        candidates: set[float] = {base_bpm}
        for _ in range(2):
            snapshot = list(candidates)
            for val in snapshot:
                if val * 2 <= 480.0:
                    candidates.add(val * 2)
                if val / 2 >= 15.0:
                    candidates.add(val / 2)
        weighted = sorted({c for c in candidates if math.isfinite(c) and c > 0})
        if weighted:
            best = min(weighted, key=lambda x: (abs(x - default), abs(x - base_bpm)))
            if abs(best - default) < abs(base_bpm - default) or math.isclose(best, default, rel_tol=1e-3):
                bpm = best
            else:
                bpm = base_bpm
        buckets = [60.0, 75.0, 90.0, 105.0, 120.0, 135.0, 150.0, 180.0]
        tol = 0.05
        for base in buckets:
            if abs(bpm - base) / base <= tol:
                bpm = float(base)
                break

    _safe_read_bpm.last_source = source  # type: ignore[attr-defined]
    return float(bpm)


def _fold_to_range(bpm: float, lo: float, hi: float) -> float | None:
    """Fold *bpm* by factors of two into ``[lo, hi]`` if possible."""

    candidates = [bpm]
    for _ in range(2):
        candidates += [c * 2 for c in candidates] + [c / 2 for c in candidates]
    valid = [c for c in candidates if lo <= c <= hi]
    if not valid:
        return None
    return min(valid, key=lambda x: abs(x - bpm))


def _resolve_tempo(
    pm: pretty_midi.PrettyMIDI,
    *,
    tempo_policy: str,
    fallback_bpm: float,
    min_bpm: float,
    max_bpm: float,
    fold_halves: bool,
    has_explicit_tempo: bool | None = None,
) -> tuple[float | None, str]:
    """Resolve tempo according to *tempo_policy*.

    Returns ``(bpm, reason)`` where ``bpm`` may be ``None`` when the policy is
    ``skip``. ``_resolve_tempo.last_source`` records the tempo source
    (``pretty_midi`` or ``mido``).
    """

    warn_on_accept = tempo_policy == "accept_warn"
    policy = tempo_policy if tempo_policy != "accept_warn" else "accept"
    inferred_default = False
    bpm: float | None = None
    source = "unknown"
    _times, tempi = pm.get_tempo_changes()
    if len(tempi) and math.isfinite(tempi[0]):
        bpm = float(tempi[0])
        source = "pretty_midi"
        if has_explicit_tempo is False:
            inferred_default = True
    elif mido is not None:  # pragma: no branch - optional dependency
        try:
            midi = pm_to_mido(pm)
            for track in midi.tracks:
                for msg in track:
                    if msg.type == "set_tempo":
                        bpm = mido.tempo2bpm(msg.tempo)
                        source = "mido"
                        break
                if bpm is not None:
                    break
        except Exception:  # pragma: no cover - failed conversion
            pass

    orig_bpm = bpm
    reason = "accept"
    if bpm is not None and fold_halves:
        folded = _fold_to_range(bpm, min_bpm, max_bpm)
        if folded is not None and abs(folded - bpm) > 1e-6:
            bpm = folded
            reason = "fold"

    invalid = (
        bpm is None
        or not math.isfinite(bpm)
        or bpm <= 0
        or bpm < min_bpm
        or bpm > max_bpm
        or inferred_default
    )
    _resolve_tempo.last_invalid = orig_bpm  # type: ignore[attr-defined]

    if invalid:
        if policy == "skip":
            _resolve_tempo.last_source = source  # type: ignore[attr-defined]
            return None, "invalid"
        if policy == "fallback":
            _resolve_tempo.last_source = "fallback"  # type: ignore[attr-defined]
            return float(fallback_bpm), f"fallback:{orig_bpm}"
        if warn_on_accept:
            _resolve_tempo.last_source = "fallback"  # type: ignore[attr-defined]
            return float(fallback_bpm), "accept_warn"
        _resolve_tempo.last_source = "fallback"  # type: ignore[attr-defined]
        return float(fallback_bpm), "accept"

    _resolve_tempo.last_source = source  # type: ignore[attr-defined]
    return float(bpm), reason


def _iter_loops(root: Path, include_audio: bool = True) -> Iterable[Path]:
    """Yield loop files under *root* with minimal traversal overhead."""

    stack = [Path(root)]
    exts = {".mid", ".midi"}
    if include_audio:
        exts |= {".wav", ".wave"}
    while stack:
        directory = stack.pop()
        for entry in directory.iterdir():
            if entry.is_dir():
                stack.append(entry)
            elif entry.suffix.lower() in exts:
                yield entry


# joblib/loky workers require top-level callables on Windows; keep arguments
# serialisable via the small ``_LoadWorkerConfig`` dataclass defined above.
def _load_worker(args: tuple[Path, _LoadWorkerConfig]) -> _LoadResult:
    path, cfg = args
    audio_failed = False
    tempo_error: str | None = None
    tempo_injected = False
    total_seconds = 0.0
    midi_obj = None
    raw_midi_obj = None
    raw_has_tempo: bool | None = None
    try:
        if path.suffix.lower() in {".wav", ".wave"}:
            pm = convert_wav_to_midi(path, fixed_bpm=cfg.fixed_bpm)
            if pm is None:
                audio_failed = True
                return _LoadResult(
                    None,
                    "error",
                    audio_failed,
                    None,
                    "error",
                    "unknown",
                    False,
                    tempo_error,
                    None,
                )
        else:
            pm = pretty_midi.PrettyMIDI(str(path))
    except Exception as exc:
        if path.suffix.lower() in {".wav", ".wave"}:
            audio_failed = True
        return _LoadResult(None, "error", audio_failed, None, "error", "unknown", False, str(exc), None)

    if cfg.inject_default_tempo > 0 and path.suffix.lower() in {".mid", ".midi"}:
        # Prefer non-destructive injection: record the desired tempo and update the
        # PrettyMIDI instance without writing the original file back to disk.
        try:
            import mido as _mido  # type: ignore

            mid = _mido.MidiFile(str(path))
            default_stub = False
            tempo_message_injected = False
            tempo_changed = False
            tempo_was_injected = False
            inject_tempo_val: int | None = None
            if cfg.inject_default_tempo > 0:
                try:
                    inject_tempo_val = int(_mido.bpm2tempo(float(cfg.inject_default_tempo)))
                except Exception:
                    inject_tempo_val = None
            tempo_slots: list[tuple[int, int]] = []
            for track_idx, track in enumerate(mid.tracks):
                for msg_idx, msg in enumerate(track):
                    if getattr(msg, "type", "") == "set_tempo":
                        tempo_slots.append((track_idx, msg_idx))
            if tempo_slots and cfg.inject_default_tempo > 0:
                default_tempos = {
                    int(_mido.bpm2tempo(120.0)),
                }
                if inject_tempo_val is not None:
                    default_tempos.add(inject_tempo_val)
                if len(tempo_slots) == 1:
                    track_idx, msg_idx = tempo_slots[0]
                    tempo_msg = mid.tracks[track_idx][msg_idx]
                    tempo_val = getattr(tempo_msg, "tempo", None)
                    if tempo_val is not None and int(tempo_val) in default_tempos:
                        if getattr(tempo_msg, "time", 0) == 0:
                            default_stub = True
                            tempo_changed = True
                            del mid.tracks[track_idx][msg_idx]
            has_explicit = any(
                getattr(msg, "type", "") == "set_tempo" for tr in mid.tracks for msg in tr
            )
            if not has_explicit and inject_tempo_val is not None and getattr(mid, "tracks", None):
                try:
                    tempo_msg = _mido.MetaMessage("set_tempo", tempo=inject_tempo_val, time=0)
                    mid.tracks[0].insert(0, tempo_msg)
                    tempo_message_injected = True
                    tempo_changed = True
                    tempo_was_injected = True
                    has_explicit = True
                except Exception:
                    pass
            if not has_explicit and inject_tempo_val is not None:
                try:
                    if not getattr(mid, "tracks", None):
                        track = _mido.MidiTrack()
                        mid.tracks.append(track)
                        tempo_changed = True
                    if getattr(mid, "tracks", None):
                        tempo_msg = _mido.MetaMessage("set_tempo", tempo=inject_tempo_val, time=0)
                        mid.tracks[0].insert(0, tempo_msg)
                        tempo_message_injected = True
                        tempo_changed = True
                        tempo_was_injected = True
                        has_explicit = True
                except Exception:
                    pass
            if tempo_changed and getattr(mid, "tracks", None):
                try:
                    marker_exists = any(
                        getattr(msg, "type", "") == "text"
                        and getattr(msg, "text", "") == _NO_TEMPO_MARKER
                        for track in mid.tracks
                        for msg in track
                    )
                except Exception:
                    marker_exists = True
            else:
                marker_exists = True
            if tempo_changed and not marker_exists and getattr(mid, "tracks", None):
                try:
                    marker_msg = _mido.MetaMessage("text", text=_NO_TEMPO_MARKER, time=0)
                    mid.tracks[0].insert(0, marker_msg)
                except Exception:
                    pass
            if tempo_changed:
                try:
                    mid.save(str(path))
                except Exception as exc:
                    tempo_error = str(exc)
                else:
                    reloaded_pm = None
                    try:
                        reloaded_pm = pretty_midi.PrettyMIDI(str(path))
                    except Exception:
                        reloaded_pm = None
                    if reloaded_pm is not None and tempo_message_injected:
                        try:
                            _rtimes, rtempi = reloaded_pm.get_tempo_changes()
                        except Exception:
                            rtempi = []
                        rtempi_size = getattr(rtempi, "size", None)
                        if rtempi_size is None:
                            try:
                                rtempi_size = len(rtempi)
                            except TypeError:
                                rtempi_size = 0
                        if rtempi_size == 0 and inject_tempo_val is not None:
                            try:
                                if not getattr(mid, "tracks", None):
                                    track = _mido.MidiTrack()
                                    mid.tracks.append(track)
                                track0 = mid.tracks[0]
                                for idx in range(len(track0) - 1, -1, -1):
                                    if getattr(track0[idx], "type", "") == "set_tempo":
                                        del track0[idx]
                                tempo_msg = _mido.MetaMessage(
                                    "set_tempo", tempo=inject_tempo_val, time=0
                                )
                                track0.insert(0, tempo_msg)
                                mid.save(str(path))
                                reloaded_pm = pretty_midi.PrettyMIDI(str(path))
                            except Exception as exc:
                                tempo_error = str(exc)
                                reloaded_pm = None
                    if reloaded_pm is not None:
                        pm = reloaded_pm
            raw_midi_obj = mid
            raw_has_tempo = has_explicit
            if tempo_was_injected:
                tempo_injected = True
                pm._composer2_injected_tempo = True  # type: ignore[attr-defined]
                try:
                    scale = 60.0 / (float(cfg.inject_default_tempo) * pm.resolution)
                    if hasattr(pm, "_tick_scales"):
                        pm._tick_scales = [(0, float(scale))]  # type: ignore[attr-defined]
                    tick_attrs = ("_PrettyMIDI__tick_to_time", "_tick_to_time")
                    for attr in tick_attrs:
                        if hasattr(pm, attr):
                            setattr(pm, attr, [0.0])
                            break
                    update_fn = getattr(pm, "_update_tick_to_time", None)
                    if callable(update_fn):
                        update_fn(pm.resolution)
                except Exception:
                    pass
        except Exception as exc:  # pragma: no cover - best effort logging upstream
            tempo_error = str(exc)

    if (
        cfg.inject_default_tempo > 0
        and path.suffix.lower() in {".mid", ".midi"}
        and mido is not None
    ):
        try:
            verify_mid = mido.MidiFile(filename=str(path))
        except Exception:
            verify_mid = None
        if verify_mid is not None:
            has_tempo_meta = any(
                getattr(msg, "type", "") == "set_tempo"
                for track in getattr(verify_mid, "tracks", [])
                for msg in track
            )
            if not has_tempo_meta:
                try:
                    tempo_val = int(mido.bpm2tempo(float(cfg.inject_default_tempo)))
                except Exception:
                    tempo_val = None
                if tempo_val is not None:
                    try:
                        if not getattr(verify_mid, "tracks", None):
                            verify_mid.tracks.append(mido.MidiTrack())
                        track0 = verify_mid.tracks[0]
                        tempo_msg = mido.MetaMessage("set_tempo", tempo=tempo_val, time=0)
                        track0.insert(0, tempo_msg)
                        verify_mid.save(str(path))
                        tempo_injected = True
                        raw_midi_obj = verify_mid
                        raw_has_tempo = True
                        try:
                            pm = pretty_midi.PrettyMIDI(str(path))
                        except Exception:
                            pass
                    except Exception as exc:  # pragma: no cover - rare fallback failures
                        tempo_error = str(exc)

    if (
        raw_has_tempo is None
        and mido is not None
        and path.suffix.lower() in {".mid", ".midi"}
    ):
        try:
            raw_midi_obj = mido.MidiFile(filename=str(path))
            raw_has_tempo = any(
                getattr(msg, "type", "") == "set_tempo" for tr in raw_midi_obj.tracks for msg in tr
            )
        except Exception:
            raw_midi_obj = None
            raw_has_tempo = None

    tempo, tempo_reason = _resolve_tempo(
        pm,
        tempo_policy=cfg.tempo_policy,
        fallback_bpm=cfg.fallback_bpm,
        min_bpm=cfg.min_bpm,
        max_bpm=cfg.max_bpm,
        fold_halves=cfg.fold_halves,
        has_explicit_tempo=raw_has_tempo,
    )
    tempo_source = getattr(_resolve_tempo, "last_source", "unknown")
    invalid_bpm = getattr(_resolve_tempo, "last_invalid", None)
    if tempo is None:
        return _LoadResult(
            None,
            "skip",
            audio_failed,
            None,
            tempo_reason,
            tempo_source,
            tempo_injected,
            tempo_error,
            invalid_bpm,
        )

    beats_per_bar_local = 4.0
    try:
        ts_changes = getattr(pm, "time_signature_changes", [])
    except Exception:
        ts_changes = []
    if ts_changes:
        ts0 = ts_changes[0]
        try:
            beats_per_bar_local = float(ts0.numerator) * (4.0 / float(ts0.denominator))
        except Exception:
            beats_per_bar_local = 4.0
    try:
        total_seconds = float(pm.get_end_time())
    except Exception:
        total_seconds = 0.0
    if not isfinite(total_seconds) or total_seconds < 0:
        total_seconds = 0.0
    try:
        midi_data = getattr(pm, "midi_data", None)
        if midi_data is not None:
            total_seconds = max(total_seconds, float(getattr(midi_data, "length", 0.0)))
    except Exception:
        pass
    try:
        latest_note = max(
            (note.end for inst in pm.instruments for note in inst.notes),
            default=0.0,
        )
    except Exception:
        latest_note = 0.0
    if latest_note > total_seconds:
        total_seconds = float(latest_note)
    if mido is not None:
        lengths: list[float] = []
        try:
            midi_obj = pm_to_mido(pm)
        except Exception:
            midi_obj = None
        if midi_obj is not None:
            try:
                cand = float(getattr(midi_obj, "length", 0.0))
                lengths.append(cand)
            except Exception:
                pass
        raw_for_length = raw_midi_obj
        if raw_for_length is None and path.suffix.lower() in {".mid", ".midi"}:
            try:
                raw_for_length = mido.MidiFile(filename=str(path))
            except Exception:
                raw_for_length = None
        if raw_for_length is not None:
            try:
                cand = float(getattr(raw_for_length, "length", 0.0))
                lengths.append(cand)
            except Exception:
                pass
        for cand in lengths:
            if math.isfinite(cand) and cand > total_seconds:
                total_seconds = cand

    if not math.isfinite(beats_per_bar_local) or beats_per_bar_local <= 0:
        beats_per_bar_local = 4.0

    def _total_beats_from_tempo_map(pm_obj, end_sec: float) -> float | None:
        try:
            times, bpms = pm_obj.get_tempo_changes()
            if not len(bpms):
                return None
            beats = 0.0
            for i, bpm in enumerate(bpms):
                t0 = float(times[i])
                t1 = float(times[i + 1]) if i + 1 < len(times) else float(end_sec)
                if t1 > t0 and math.isfinite(bpm) and bpm > 0:
                    beats += (t1 - t0) * (bpm / 60.0)
            return beats
        except Exception:
            return None

    if not isfinite(total_seconds) or total_seconds <= 0:
        total_seconds = 0.0

    total_beats = _total_beats_from_tempo_map(pm, total_seconds)
    if total_beats is None or total_beats <= 0:
        try:
            beat_times = pm.get_beats()
            if beat_times is not None and len(beat_times) > 0:
                total_beats = float(len(beat_times))
                logger.debug(
                    "[groove loader] beat fallback provided %s beats for %s",
                    total_beats,
                    path.name,
                )
        except Exception:
            pass
    if (
        (total_beats is None or total_beats <= 0)
        and tempo
        and math.isfinite(tempo)
        and tempo > 0
        and total_seconds > 0
    ):
        total_beats = (tempo / 60.0) * total_seconds

    bars = (total_beats / beats_per_bar_local) if (total_beats and total_beats > 0) else 0.0

    allow_drums = True if cfg.allow_drums is None else bool(cfg.allow_drums)
    allow_pitched = True if cfg.allow_pitched is None else bool(cfg.allow_pitched)
    instrument_whitelist = cfg.instrument_whitelist if cfg.instrument_whitelist else None
    logger.debug(
        "[groove loader] effective filters for %s: allow_drums=%s allow_pitched=%s",
        path.name,
        allow_drums,
        allow_pitched,
    )

    def _fallback_events_from_mido() -> dict[str, object] | None:
        """Derive note events via :mod:`mido` when PrettyMIDI yields nothing."""

        if mido is None:
            return None

        nonlocal midi_obj

        local_midi = midi_obj
        if local_midi is None:
            try:
                local_midi = mido.MidiFile(filename=str(path))
            except Exception:
                return None

        ticks_per_beat = float(getattr(local_midi, "ticks_per_beat", 480) or 480)
        starts: dict[tuple[int, int], list[int]] = {}
        events: list[tuple[float, int]] = []
        uniq: set[int] = set()
        max_tick = 0
        allowed_notes = 0
        has_drum = False
        program_by_channel: dict[int, int | None] = {}

        def _close(channel: int, note: int) -> None:
            nonlocal allowed_notes, has_drum
            key = (channel, note)
            start_list = starts.get(key)
            if not start_list:
                return
            start_tick = start_list.pop(0)
            beat_pos = start_tick / ticks_per_beat
            events.append((beat_pos, note))
            uniq.add(note)
            drum = channel == 9
            if drum:
                has_drum = True
            if _instrument_allowed(
                program_by_channel.get(channel), drum, instrument_whitelist
            ) and ((drum and allow_drums) or ((not drum) and allow_pitched)):
                allowed_notes += 1

        for track in getattr(local_midi, "tracks", []):
            tick = 0
            for msg in track:
                tick += int(getattr(msg, "time", 0) or 0)
                msg_type = getattr(msg, "type", None)
                if msg_type == "note_on":
                    channel = int(getattr(msg, "channel", 0) or 0)
                    note = int(getattr(msg, "note", 0) or 0)
                    velocity = int(getattr(msg, "velocity", 0) or 0)
                    if velocity > 0:
                        starts.setdefault((channel, note), []).append(tick)
                    else:
                        _close(channel, note)
                elif msg_type == "note_off":
                    channel = int(getattr(msg, "channel", 0) or 0)
                    note = int(getattr(msg, "note", 0) or 0)
                    _close(channel, note)
                elif msg_type == "program_change":
                    channel = int(getattr(msg, "channel", 0) or 0)
                    program = getattr(msg, "program", None)
                    program_by_channel[channel] = int(program) if program is not None else None
                max_tick = max(max_tick, tick)

        if not events:
            return None

        events.sort(key=lambda x: x[0])
        if allowed_notes <= 0:
            allowed_notes = len(events)

        bars_fb = 0.0
        if beats_per_bar_local > 0:
            bars_fb = (max_tick / ticks_per_beat) / float(beats_per_bar_local)

        midi_obj = local_midi
        return {
            "events": events,
            "note_count": allowed_notes,
            "uniq_pitches": len(uniq) if uniq else len({p for _, p in events}),
            "has_drum": has_drum,
            "bars": bars_fb,
        }

    raw_note_count = 0
    for inst in pm.instruments:
        if not _instrument_allowed(inst.program, inst.is_drum, instrument_whitelist):
            continue
        if (inst.is_drum and allow_drums) or (not inst.is_drum and allow_pitched):
            raw_note_count += len(inst.notes)

    min_bars = float(cfg.min_bars)
    min_notes = int(cfg.min_notes)
    notes = midi_to_events(pm, tempo)
    note_cnt = len(notes)
    uniq_pitches = len({pitch for _, pitch in notes}) if notes else 0
    is_drum = any(
        inst.is_drum
        for inst in pm.instruments
        if _instrument_allowed(inst.program, inst.is_drum, instrument_whitelist)
    )

    fallback_data = None
    need_fallback = (not notes or raw_note_count == 0)
    if not need_fallback and mido is not None and min_bars > 0 and bars < min_bars:
        need_fallback = True
    if need_fallback and mido is not None:
        fallback_data = _fallback_events_from_mido()
        if fallback_data is not None:
            if not notes:
                notes = list(fallback_data["events"])
                note_cnt = len(notes)
                uniq_pitches = fallback_data.get("uniq_pitches", uniq_pitches)
            if raw_note_count == 0:
                raw_note_count = int(fallback_data.get("note_count", raw_note_count))
            fb_bars = float(fallback_data.get("bars", 0.0))
            if fb_bars > 0.0 and (not math.isfinite(bars) or bars <= 0.0 or fb_bars > bars):
                bars = fb_bars
            if not is_drum and bool(fallback_data.get("has_drum", False)):
                is_drum = True

    raw_note_count = max(raw_note_count, note_cnt)

    if not math.isfinite(bars) or bars < 0.0:
        bars = 0.0

    short_bars = min_bars > 0 and bars < min_bars
    few_notes = min_notes > 0 and raw_note_count < min_notes
    try:
        logger.debug(
            "[groove loader] %s summary: bars=%.3f notes=%d uniq=%d drum=%s fallback=%s short_bars=%s few_notes=%s",
            path.name,
            bars,
            raw_note_count,
            uniq_pitches,
            is_drum,
            fallback_data is not None,
            short_bars,
            few_notes,
        )
    except Exception:
        pass

    if not notes:
        return _LoadResult(
            None,
            "notes",
            audio_failed,
            tempo,
            tempo_reason,
            tempo_source,
            tempo_injected,
            tempo_error,
            invalid_bpm,
        )

    try:
        notes.sort(key=lambda x: (x[0], x[1]))
    except Exception:
        pass
    offs = [off for off, _ in notes]

    note_cnt = len(notes)
    uniq_pitches = len({pitch for _, pitch in notes})
    if fallback_data is not None and uniq_pitches == 0:
        uniq_pitches = int(fallback_data.get("uniq_pitches", 0))
    is_drum = is_drum or any(
        inst.is_drum
        for inst in pm.instruments
        if _instrument_allowed(inst.program, inst.is_drum, instrument_whitelist)
    )
    name_tokens = [tok for tok in re.split(r"[^a-z0-9]+", path.stem.lower()) if tok]
    has_fill_token = "fill" in name_tokens
    is_fill = has_fill_token

    logger.debug(
        "accept: %s bars=%.2f notes=%d (beats/bar=%.2f tempo=%s)",
        path,
        bars,
        note_cnt,
        float(beats_per_bar_local),
        str(tempo),
    )

    payload = (notes, offs, bars, note_cnt, uniq_pitches, is_drum, is_fill)
    return _LoadResult(
        payload,
        "ok",
        audio_failed,
        tempo,
        tempo_reason,
        tempo_source,
        tempo_injected,
        tempo_error,
        invalid_bpm,
    )


# 上限（大規模ディレクトリ時の探索打切り）。環境変数で調整可。
_DEFAULT_FILE_BUDGET = int(os.getenv("GROOVE_SAMPLER_FILE_BUDGET", "64"))

def train(
    loop_dir: Path,
    *,
    n: int = 4,
    auto_res: bool = False,
    coarse: bool = False,
    beats_per_bar: int | None = None,
    n_jobs: int | None = None,
    memmap_dir: Path | None = None,
    fixed_bpm: float | None = None,
    progress: bool = False,
    include_audio: bool = True,
    aux_key: str | None = None,
    tempo_policy: str = "fallback",
    fallback_bpm: float = 120.0,
    min_bpm: float = 40.0,
    max_bpm: float = 300.0,
    fold_halves: bool = False,
    tempo_verbose: bool = False,
    min_bars: float = 1.0,
    min_notes: int = 8,
    drum_only: bool = False,
    pitched_only: bool = False,
    tag_fill_from_filename: bool = True,
    exclude_fills: bool = False,
    separate_fills: bool = False,
    len_sampling: str = "sqrt",
    inject_default_tempo: float = 0.0,
    snapshot_interval: int = 0,
    counts_dtype: str = "uint32",
    hash_buckets: int = 1 << 20,
    store_backend: str = "sqlite",
    db_path: Path | None = None,
    commit_every: int = 2000,
    db_busy_timeout_ms: int = 60000,
    db_synchronous: str = "NORMAL",
    db_mmap_mb: int = 64,
    dedup_filter: str = "cuckoo" if _HAS_CUCKOO else "sqlite",
    max_ram_mb: int = 0,
    min_count: int = 2,
    prune_interval: int = 200000,
    flush_interval: int = 50000,
    train_mode: str = "stream",
    max_rows_per_shard: int = 1_000_000,
    resume: bool = False,
    aux_vocab_path: Path | None = None,
    cache_probs_memmap: bool = False,
    instrument_whitelist: Iterable[int] | None = None,
):
    """Build a hashed n‑gram model from drum loops located in *loop_dir*."""
    if mido is None:  # pragma: no cover - dependency is missing
        raise RuntimeError(
            "mido is required for groove sampler training; install via 'pip install mido'"
        )

    requested_inject_tempo = inject_default_tempo
    forced_relax_tempo: float | None = None
    relax_filters = os.getenv("TEST_RELAX_FILTERS") == "1"
    if relax_filters:
        min_bars = min(min_bars, 1.0)
        min_notes = min(min_notes, 1)
        # Tempo injection priority: explicit CLI value > relax-forced 100 BPM > fallback BPM.
        if inject_default_tempo <= 0:
            forced_relax_tempo = 100.0 if fallback_bpm != 100.0 else fallback_bpm
            inject_default_tempo = forced_relax_tempo

    tempo_defaults_meta = {
        "policy": tempo_policy,
        "fallback_bpm": fallback_bpm,
        "requested_inject": requested_inject_tempo,
        "effective_inject": inject_default_tempo,
        "relax_forced": forced_relax_tempo,
    }

    if n_jobs is None:
        n_jobs = min(4, os.cpu_count() or 1)

    memmap_dir_given = memmap_dir is not None
    if memmap_dir is None:
        memmap_dir = Path(tempfile.gettempdir()) / "composer2_groove_mm"

    # 先に resume ログを読む（列挙中にスキップしたいので）
    processed_log = memmap_dir / "processed.txt"
    processed_set: set[str] = (
        set(processed_log.read_text().splitlines())
        if (resume and processed_log.exists())
        else set()
    )
    if processed_set:
        logger.info("resume: %d files previously processed", len(processed_set))

    if counts_dtype in {"u32", "uint32"}:
        counts_dtype = "uint32"
    elif counts_dtype in {"u64", "uint64"}:
        counts_dtype = "uint64"
    else:
        raise ValueError("counts_dtype must be 'u32' or 'u64'")

    # 大量ファイル時も探索を O(budget) で打ち切る（resume は列挙中に反映）
    file_budget = _DEFAULT_FILE_BUDGET
    try:
        env_budget = int(os.getenv("GROOVE_SAMPLER_FILE_BUDGET", str(file_budget)))
        if env_budget > 0:
            file_budget = env_budget
    except Exception:
        pass

    paths: list[Path] = []
    for p in _iter_loops(loop_dir, include_audio):
        if resume and str(p) in processed_set:
            continue
        paths.append(p)
        if file_budget > 0 and len(paths) >= file_budget:
            break

    model_path = memmap_dir / "model.pkl"
    if not paths:
        if resume and model_path.exists():
            return load(model_path, aux_vocab_path)
        raise SystemExit("No files found — training aborted")

    paths = [p for p in paths if str(p) not in processed_set]
    if aux_vocab_path and aux_vocab_path.exists():
        aux_vocab_obj = AuxVocab.from_json(aux_vocab_path)
    else:
        aux_vocab_obj = AuxVocab()
    total_events = 0
    files_skipped = 0           # fatal skips only (e.g., audio decode failure)
    files_filtered = 0          # filtered out by min_notes, etc.
    tempo_stats = {"accept": 0, "accept_warn": 0, "fold": 0, "fallback": 0, "skip": 0}
    skipped_paths: list[Path] = []

    def _empty_model(resolution_hint: int = 16) -> NGramModel:
        res = int(resolution_hint) if resolution_hint else 16
        res = max(res, 1)
        res_coarse = res // 4 if coarse else res
        model = NGramModel(
            n=n,
            resolution=res,
            resolution_coarse=res_coarse,
            state_to_idx={},
            idx_to_state=[],
            freq=[{} for _ in range(n)],
            bucket_freq={},
            ctx_maps=[{} for _ in range(n)],
            prob_paths=None,
            prob=None,
            aux_vocab=aux_vocab_obj,
            version=2,
            file_weights=[],
            files_scanned=len(paths),
            files_skipped=files_skipped,
            total_events=0,
            hash_buckets=hash_buckets,
            tempo_defaults=tempo_defaults_meta,
        )
        model.files_skipped_fatal = files_skipped
        model.files_filtered = files_filtered
        return model

    if instrument_whitelist:
        whitelist: set[int] | None = set(instrument_whitelist)
    else:
        whitelist = None

    if drum_only and pitched_only:
        logger.warning(
            "Both --drums-only and --pitched-only were set; treating as 'allow both'."
        )
        allow_drums_flag: bool | None = True
        allow_pitched_flag: bool | None = True
    elif drum_only:
        allow_drums_flag = True
        allow_pitched_flag = False
    elif pitched_only:
        allow_drums_flag = False
        allow_pitched_flag = True
    else:
        allow_drums_flag = None
        allow_pitched_flag = None

    worker_cfg = _LoadWorkerConfig(
        fixed_bpm=fixed_bpm,
        inject_default_tempo=inject_default_tempo,
        tempo_policy=tempo_policy,
        fallback_bpm=fallback_bpm,
        min_bpm=min_bpm,
        max_bpm=max_bpm,
        fold_halves=fold_halves,
        tag_fill_from_filename=tag_fill_from_filename,
        allow_drums=allow_drums_flag,
        allow_pitched=allow_pitched_flag,
        min_bars=float(min_bars),
        min_notes=int(min_notes),
        instrument_whitelist=whitelist,
    )

    # Allow environment override to avoid multiprocessing on constrained systems.
    if n_jobs is None:
        try:
            env_n = os.getenv("GROOVE_N_JOBS") or os.getenv("JOBLIB_N_JOBS")
            if env_n is not None:
                n_jobs = int(env_n)
        except Exception:
            n_jobs = None
    if n_jobs == 1 or n_jobs == 0:
        loaded = [_load_worker((p, worker_cfg)) for p in paths]
    else:
        try:
            loaded = Parallel(n_jobs=n_jobs)(
                delayed(_load_worker)((p, worker_cfg)) for p in paths
            )
        except Exception as exc:
            logger.warning(
                "Parallel execution unavailable (%s); falling back to sequential", exc
            )
            loaded = [_load_worker((p, worker_cfg)) for p in paths]

    aux_values: list[str | None] = []
    results: list[tuple[list[tuple[float, int]], list[float]]] = []
    bars_list: list[float] = []
    reason_counts: dict[str, int] = {
        "tempo_skip": 0,
        "bars": 0,
        "notes": 0,
        "fill": 0,
        "drum_only": 0,
        "pitched_only": 0,
        "error": 0,
    }

    def _bump_reason(key: str) -> None:
        reason_counts[key] = reason_counts.get(key, 0) + 1

    fallback_candidate: tuple[list[tuple[float, int]], list[float]] | None = None
    fallback_bars: float | None = None
    fallback_aux: str | None = None
    fallback_path: Path | None = None
    fallback_reason: str | None = None
    for p, res in zip(paths, loaded):
        if res.reason == "error":
            if res.tempo_error:
                logger.warning("Failed to load %s: %s", p, res.tempo_error)
            suffix = p.suffix.lower()
            audio_issue = res.audio_failed or suffix in {".wav", ".wave"}
            if audio_issue:
                skipped_paths.append(p)
                files_skipped += 1
            else:
                files_filtered += 1
                _bump_reason("error")
            continue

        if res.tempo_error:
            logger.warning("Failed to ensure tempo for %s: %s", p, res.tempo_error)
        if res.tempo_injected:
            logger.warning("Injected tempo %.2f BPM for %s", worker_cfg.inject_default_tempo, p)

        if res.tempo is None:
            tempo_stats["skip"] += 1
            skipped_paths.append(p)
            files_filtered += 1
            _bump_reason("tempo_skip")
            logger.warning("Skipping %s due to %s tempo", p, res.tempo_reason)
            continue

        if res.tempo_reason == "accept_warn":
            tempo_stats["accept_warn"] += 1
        elif res.tempo_reason.startswith("fallback"):
            tempo_stats["fallback"] += 1
        elif res.tempo_reason == "fold":
            tempo_stats["fold"] += 1
        else:
            tempo_stats["accept"] += 1

        if res.tempo_reason == "accept_warn":
            bad = res.invalid_bpm if res.invalid_bpm is not None else "unknown"
            logger.warning(
                "Invalid tempo %s in %s; using fallback %.2f BPM",
                bad,
                p,
                res.tempo,
            )
        elif res.tempo_source == "fallback" and res.tempo_reason == "accept":
            logger.warning("Invalid tempo in %s; using fallback %.2f BPM", p, res.tempo)
        else:
            logger.info("Tempo=%.2f via %s for %s", res.tempo, res.tempo_source, p)

        payload = res.payload
        audio_decode_failed = res.audio_failed
        load_reason = res.reason
        if payload is None:
            suffix = p.suffix.lower()
            audio_issue = audio_decode_failed or suffix in {".wav", ".wave"}
            if audio_issue:
                skipped_paths.append(p)
                files_skipped += 1
            else:
                files_filtered += 1
                if load_reason:
                    if load_reason == "error":
                        _bump_reason("error")
                    elif load_reason == "skip":
                        _bump_reason("tempo_skip")
                    else:
                        _bump_reason(load_reason)
            continue
        notes, offs, bars, note_cnt, uniq_pitches, is_drum, is_fill = payload
        raw_is_fill = bool(is_fill)
        is_fill_effective = raw_is_fill if (tag_fill_from_filename or exclude_fills) else False
        reason: str | None = None
        if bars < min_bars:
            reason = "bars"
        elif note_cnt < min_notes:
            reason = "notes"
        elif drum_only and not is_drum:
            reason = "drum_only"
        elif pitched_only and is_drum:
            reason = "pitched_only"
        elif exclude_fills and is_fill_effective:
            reason = "fill"

        if relax_filters and reason == "notes" and note_cnt == 0:
            reason = None

        if reason is not None:
            if (
                reason in {"notes", "bars"}
                and note_cnt > 0
                and fallback_candidate is None
            ):
                meta = load_meta(p)
                fallback_candidate = (notes, offs)
                fallback_bars = bars
                fallback_aux = meta.get(aux_key) if aux_key else None
                fallback_path = p
                fallback_reason = reason
            skipped_paths.append(p)
            _bump_reason(reason)
            files_filtered += 1
            continue

        if audio_decode_failed:
            skipped_paths.append(p)
            files_skipped += 1
            continue

        results.append((notes, offs))
        bars_list.append(bars)
        meta = load_meta(p)
        aux_values.append(meta.get(aux_key) if aux_key else None)

    if tempo_verbose:
        logger.info(
            "tempo summary: total=%d accept=%d warn=%d fold=%d fallback=%d skip=%d",
            len(paths),
            tempo_stats["accept"],
            tempo_stats["accept_warn"],
            tempo_stats["fold"],
            tempo_stats["fallback"],
            tempo_stats["skip"],
        )
        summary_items = ", ".join(f"{k}={v}" for k, v in reason_counts.items())
        logger.info("filter summary: %s", summary_items if summary_items else "none")
        if skipped_paths:
            preview = ", ".join(str(p) for p in skipped_paths[:20])
            if len(skipped_paths) > 20:
                preview += f", ... and {len(skipped_paths) - 20} more"
            logger.info("skipped files: %s", preview)

    if not results and fallback_candidate is not None and fallback_bars is not None:
        logger.warning(
            "No events collected with min_notes=%d; relaxing filter to keep %s",
            worker_cfg.min_notes,
            fallback_path,
        )
        results.append(fallback_candidate)
        bars_list.append(fallback_bars)
        aux_values.append(fallback_aux)
        if fallback_path is not None:
            try:
                skipped_paths.remove(fallback_path)
            except ValueError:
                pass
        if fallback_reason:
            count = reason_counts.get(fallback_reason, 0)
            if count > 0:
                reason_counts[fallback_reason] = count - 1
        if files_filtered > 0:
            files_filtered -= 1

    if not results:
        if relax_filters:
            logger.info(
                "relaxed filters active; returning empty model with zero events"
            )
        logger.warning(
            "No events collected — returning empty model (files=%d skipped=%d filtered=%d)",
            len(paths),
            files_skipped + files_filtered,
            files_filtered,
        )
        model = _empty_model()
        model.files_skipped = files_skipped + files_filtered
        model.files_skipped_fatal = files_skipped
        model.files_filtered = files_filtered
        model.files_scanned = len(paths)
        save(model, model_path)
        return model

    note_seqs = [r[0] for r in results]
    all_offsets = [off for r in results for off in r[1]]

    resolution = int(infer_resolution(all_offsets, beats_per_bar=beats_per_bar) if auto_res else 16)
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

    effective_mode = train_mode
    num_files = len(paths)
    avg_events = total_events / max(1, num_files)
    if train_mode == "stream" and _is_small_corpus(total_events, num_files, avg_events):
        effective_mode = "inmemory"

    if len_sampling == "uniform":
        file_weights = [1.0 for _ in bars_list]
    elif len_sampling == "proportional":
        file_weights = [float(b) for b in bars_list]
    else:
        file_weights = [math.sqrt(max(1e-6, float(b))) for b in bars_list]

    n_states = len(idx_to_state)
    bucket_freq: dict[int, np.ndarray] = {}
    hb = hash_buckets

    freq_orders: list[FreqTable] = []
    if effective_mode == "inmemory":
        store_cls = MemoryNGramStore
        stores = [store_cls() for _ in range(n)]
        local_buffers: list[list[tuple[int, int, int, int]]] = [[] for _ in range(n)]
        for seq in seqs:
            for i, cur in enumerate(seq):
                bucket = idx_to_state[cur][1]
                b_arr = bucket_freq.get(bucket)
                if b_arr is None:
                    b_arr = np.zeros(n_states, dtype=np.uint32)
                    bucket_freq[bucket] = b_arr
                b_arr[cur] += 1
                for order in range(n):
                    if order > i:
                        break
                    ctx_seq = seq[i - order : i]
                    ctx_vals = list(ctx_seq) + [order, 0]
                    ctx_hash = _hash_ctx(ctx_vals) % hb
                    local_buffers[order].append((ctx_hash, 0, cur, 1))
        for order, st in enumerate(stores):
            buf = local_buffers[order]
            if buf:
                st.bulk_inc(buf)
            st.finalize()
        for st in stores:
            table: FreqTable = {}
            for ctx_hash, _aux, next_evt, count in getattr(st, "iter_rows")():
                arr = table.get(ctx_hash)
                if arr is None:
                    arr = np.zeros(n_states, dtype=np.uint32)
                    table[ctx_hash] = arr
                arr[next_evt] = count
            freq_orders.append(table)
    else:
        mem_store = MemmapNGramStore(
            memmap_dir,
            n_orders=n,
            vocab_size=n_states,
            hash_buckets=hb,
            dtype=counts_dtype,
            mode="r+" if resume else "w+",
        )
        tables: list[FreqTable] = [dict() for _ in range(n)]
        events_seen = 0
        try:
            for seq in seqs:
                for i, cur in enumerate(seq):
                    bucket = idx_to_state[cur][1]
                    b_arr = bucket_freq.get(bucket)
                    if b_arr is None:
                        b_arr = np.zeros(n_states, dtype=np.uint32)
                        bucket_freq[bucket] = b_arr
                    b_arr[cur] += 1
                    for order in range(n):
                        if order > i:
                            break
                        ctx_seq = seq[i - order : i]
                        ctx_vals = list(ctx_seq) + [order, 0]
                        ctx_hash = _hash_ctx(ctx_vals) % hb
                        bump_count(tables[order], ctx_hash, cur, n_states)
                    events_seen += 1
                    if events_seen % flush_interval == 0:
                        mem_store.flush(tables)
                        tables = [dict() for _ in range(n)]
            if any(tables[order] for order in range(n)):
                mem_store.flush(tables)
            freq_orders = mem_store.merge()
            mem_store.write_meta()
            for order, table in enumerate(freq_orders):
                logger.info("shard order %d rows=%d", order, len(table))
        finally:
            for order_maps in mem_store.maps:
                for mm in order_maps:
                    mm.flush()
                    del mm
            gc.collect()
    aux_hit = sum(1 for v in aux_values if v)
    if aux_values:
        logger.info("aux hit rate=%.2f", aux_hit / len(aux_values))
    skipped_total = files_skipped + files_filtered
    logger.info(
        "Scanned %d files (skipped %d) \u2192 %d events \u2192 %d states",
        len(paths),
        skipped_total,
        total_events,
        len(idx_to_state),
    )
    for k, v in reason_counts.items():
        logger.info("  %s: %d", k, v)
    if total_events == 0 or len(idx_to_state) == 0:
        logger.warning(
            "No events collected - returning empty model (events=%d states=%d)",
            total_events,
            len(idx_to_state),
        )
        empty_model = _empty_model(resolution)
        empty_model.files_skipped = skipped_total
        empty_model.files_skipped_fatal = files_skipped
        empty_model.files_filtered = files_filtered
        empty_model.files_scanned = len(paths)
        save(empty_model, model_path)
        return empty_model
    model = NGramModel(
        n=n,
        resolution=resolution,
        resolution_coarse=resolution_coarse,
        state_to_idx=state_to_idx,
        idx_to_state=idx_to_state,
        freq=freq_orders,
        bucket_freq=bucket_freq,
        ctx_maps=[{} for _ in range(n)],
        prob_paths=None,
        prob=None,
        aux_vocab=aux_vocab_obj,
        version=2,
        file_weights=file_weights,
        files_scanned=len(paths),
        files_skipped=skipped_total,
        total_events=total_events,
        hash_buckets=hb,
        tempo_defaults=tempo_defaults_meta,
    )
    model.files_skipped_fatal = files_skipped
    model.files_filtered = files_filtered
    if (memmap_dir_given or cache_probs_memmap) and model.idx_to_state:
        _write_prob_memmaps(model, memmap_dir)
    if aux_vocab_path:
        # ensure parent directory exists before writing
        aux_vocab_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if hasattr(aux_vocab_obj, "to_json"):
                # prefer native serializer when available
                aux_vocab_obj.to_json(aux_vocab_path)
            elif hasattr(aux_vocab_obj, "id_to_str"):
                # fallback: list of strings
                data = getattr(aux_vocab_obj, "id_to_str")
                if not isinstance(data, list):
                    data = list(data)
                aux_vocab_path.write_text(json.dumps(data))
            else:
                # last resort: best-effort JSON dump with vars() fallback
                logger.warning("aux vocab compat path for %s", aux_vocab_path)
                if hasattr(aux_vocab_obj, "__dict__") and aux_vocab_obj.__dict__:
                    aux_vocab_path.write_text(
                        json.dumps(aux_vocab_obj.__dict__, default=str)
                    )
                else:
                    aux_vocab_path.write_text("[]")
        except Exception as exc:  # pragma: no cover - compatibility path
            logger.warning("failed to write aux vocab to %s: %s", aux_vocab_path, exc)

    if resume:
        save(model, model_path)
        processed_log.write_text("\n".join(processed_set | {str(p) for p in paths}) + "\n")
    return model


def train_streaming(
    paths: Iterable[Path],
    *,
    output: Path,
    min_bytes: int = 0,
    min_notes: int = 8,
    max_files: int | None = None,
    progress: bool = True,
    log_every: int = 200,
    save_every: int = 0,
    checkpoint_dir: Path | None = None,
    resume_from: Path | None = None,
    gc_every: int = 1000,
    mem_stats: bool = False,
    fail_fast: bool = False,
) -> dict[str, Any]:
    """Stream MIDI files and build simple pitch counts.

    This helper is intentionally lightweight and only tracks per-pitch note
    counts alongside basic bookkeeping statistics.  It is sufficient for large
    corpora sharding tests while keeping backward compatibility with the
    original n-gram training routine.
    """

    pitch_counts: dict[int, int] = {}
    processed = 0
    kept = 0
    skipped = 0

    if resume_from is not None and resume_from.exists():
        with resume_from.open("rb") as fh:
            state = pickle.load(fh)
        pitch_counts = state.get("counts", {})
        processed = state.get("processed", 0)
        kept = state.get("kept", 0)
        skipped = state.get("skipped", 0)
        logger.info(
            "Resuming streaming run from %s (processed=%d kept=%d skipped=%d)",
            resume_from,
            processed,
            kept,
            skipped,
        )

    path_list = list(paths)
    start_idx = processed
    iterator = path_list[start_idx:]
    if progress:
        iterator = tqdm(iterator, total=len(path_list) - start_idx)

    for idx, p in enumerate(iterator, start=start_idx + 1):
        if max_files is not None and kept >= max_files:
            break
        processed += 1
        try:
            if os.path.getsize(p) < min_bytes:
                skipped += 1
                continue
            pm = pretty_midi.PrettyMIDI(str(p))
            note_cnt = sum(len(inst.notes) for inst in pm.instruments)
            if note_cnt < min_notes:
                skipped += 1
                continue
            for inst in pm.instruments:
                for note in inst.notes:
                    pitch_counts[note.pitch] = pitch_counts.get(note.pitch, 0) + 1
            kept += 1
        except Exception as exc:  # pragma: no cover - best effort
            skipped += 1
            logger.warning("Failed to process %s: %s", p, exc)
            if fail_fast:
                raise
        if save_every and kept and kept % save_every == 0:
            ckpt_dir = checkpoint_dir or output.parent
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_dir / f"ckpt_{kept}.pkl"
            with ckpt_path.open("wb") as fh:
                pickle.dump(
                    {
                        "version": 1,
                        "counts": pitch_counts,
                        "processed": processed,
                        "kept": kept,
                        "skipped": skipped,
                    },
                    fh,
                )
            logger.info("saved checkpoint %s (processed=%d, kept=%d)", ckpt_path, processed, kept)
        if gc_every and processed % gc_every == 0:
            gc.collect()
        if log_every and processed % log_every == 0:
            if mem_stats and psutil is not None:
                rss = psutil.Process(os.getpid()).memory_info().rss // (1024 * 1024)
                logger.info(
                    "processed=%d kept=%d skip=%d rss=%dMB",
                    processed,
                    kept,
                    skipped,
                    rss,
                )
            else:
                logger.info(
                    "processed=%d kept=%d skip=%d",
                    processed,
                    kept,
                    skipped,
                )

    result = {
        "version": 1,
        "counts": pitch_counts,
        "processed": processed,
        "kept": kept,
        "skipped": skipped,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as fh:
        pickle.dump(result, fh)
    return result


def merge_streaming_models(paths: Iterable[Path], output: Path) -> dict[str, Any]:
    """Merge multiple streaming model pickles into one."""

    merged: dict[str, Any] | None = None
    for p in paths:
        with p.open("rb") as fh:
            data = pickle.load(fh)
        if merged is None:
            merged = {
                "version": data.get("version", 1),
                "counts": dict(data.get("counts", {})),
                "processed": data.get("processed", 0),
                "kept": data.get("kept", 0),
                "skipped": data.get("skipped", 0),
            }
            continue
        if merged.get("version") != data.get("version"):
            logger.warning("version mismatch when merging %s", p)
        for k, v in data.get("counts", {}).items():
            merged["counts"][k] = merged["counts"].get(k, 0) + v
        merged["processed"] += data.get("processed", 0)
        merged["kept"] += data.get("kept", 0)
        merged["skipped"] += data.get("skipped", 0)

    if merged is None:
        merged = {"version": 1, "counts": {}, "processed": 0, "kept": 0, "skipped": 0}

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as fh:
        pickle.dump(merged, fh)
    return merged


def _choose(probs: np.ndarray, rng: Random) -> int:
    """Sample an index from a normalized probability vector."""

    assert np.isclose(probs.sum(), 1.0, atol=1e-6), "probabilities must sum to 1"
    cdf = np.cumsum(probs)
    r = rng.random()
    idx = int(np.searchsorted(cdf, r, side="right"))
    if idx >= len(probs):
        idx = len(probs) - 1
    return idx


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


_MODEL_CACHE: dict[int, "NGramModel"] = {}


def _get_arr_internal(model_id: int, order: int, ctx_hash: int):
    """Retrieve frequency array for ``ctx_hash`` at given ``order``."""
    model = _MODEL_CACHE[model_id]
    arr = model.freq[order].get(ctx_hash)
    return None if arr is None else arr


if os.getenv("GROOVE_CACHE", "1") != "0":
    _get_arr = lru_cache(maxsize=8192)(_get_arr_internal)  # type: ignore
else:  # pragma: no cover - environment toggle
    _get_arr = _get_arr_internal


def _normalize_probs(probs: np.ndarray) -> np.ndarray:
    """Return a safe, normalized probability vector.

    - Coerces to finite, non‑negative floats
    - Falls back to uniform when the sum is zero
    """
    p = np.asarray(probs, dtype=np.float64)
    # replace NaN/Inf with 0
    p[~np.isfinite(p)] = 0.0
    # clamp negatives
    p[p < 0] = 0.0
    s = float(p.sum())
    if s <= 0.0:
        if p.size == 0:
            return p
        return np.full_like(p, 1.0 / float(p.size), dtype=np.float64)
    return p / s


def _resolve_aux_ids(
    aux_vocab: AuxVocab | None,
    cond: dict[str, str] | None,
    fallback: str | None,
) -> list[int]:
    """Return auxiliary ids to probe ordered by preference."""

    if aux_vocab is None:
        return [0]
    if not cond:
        return [0]

    key = "|".join(f"{k}={v}" for k, v in sorted(cond.items()))
    aux_any = 0
    aux_unknown = aux_vocab.str_to_id.get("<UNK>")
    idx = aux_vocab.str_to_id.get(key)

    if idx is not None:
        order = [idx]
        if fallback in {"allow", "prefer"} and aux_any not in order:
            order.append(aux_any)
        return order

    order: list[int] = []
    if fallback == "prefer":
        order.append(aux_any)
    if aux_unknown is not None:
        order.append(aux_unknown)
    if fallback != "prefer" and aux_any not in order:
        order.append(aux_any)

    return order or [aux_any]


def next_prob_dist(
    model: "NGramModel",
    history: list[int],
    bucket: int,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    cond: dict[str, str] | None = None,
    aux_fallback: str | None = None,
) -> np.ndarray:
    """Return probability distribution for next state."""

    _MODEL_CACHE[id(model)] = model
    n = model.n if model.n is not None else 4
    aux_vocab = getattr(model, "aux_vocab", None)
    aux_ids = _resolve_aux_ids(aux_vocab, cond, aux_fallback)
    hb = model.hash_buckets

    def _probe(order: int, ctx: list[int], aux_id: int) -> np.ndarray | None:
        table = model.freq[order] if order < len(model.freq) else None
        if not table:
            return None

        base_ctx_hash = _hash_ctx(list(ctx)) if ctx else 0
        aux_name = (
            aux_vocab.id_to_str[aux_id]
            if aux_vocab and aux_id < len(aux_vocab.id_to_str)
            else aux_id
        )

        arr = table.get((base_ctx_hash, aux_id))
        if arr is not None:
            arr = arr.astype(np.float64, copy=False)
            if np.isfinite(arr).all():
                total = float(arr.sum())
                if total > 0:
                    logger.debug("aux='%s' order=%d ctx=%s -> tuple-key hit", aux_name, order, ctx)
                    return arr

        ctx_hash = _hash_ctx(list(ctx) + [order, aux_id]) % hb
        arr = _get_arr(id(model), order, ctx_hash)
        if arr is not None:
            arr = arr.astype(np.float64, copy=False)
            if np.isfinite(arr).all():
                total = float(arr.sum())
                if total > 0:
                    logger.debug("aux='%s' order=%d ctx=%s -> hashed-bucket hit", aux_name, order, ctx)
                    return arr

        return None

    arr: np.ndarray | None = None
    for order in range(min(len(history), n - 1), 0, -1):
        ctx = history[-order:]
        for aux_id in aux_ids:
            arr = _probe(order, ctx, aux_id)
            if arr is not None:
                break
        if arr is not None:
            break
    else:
        arr = None
        for aux_id in aux_ids:
            arr = _probe(0, [], aux_id)
            if arr is not None:
                break

    if arr is None:
        b_arr = model.bucket_freq.get(bucket) if model.bucket_freq is not None else None
        if b_arr is not None:
            b_arr = b_arr.astype(np.float64, copy=False)
            if np.isfinite(b_arr).all() and float(b_arr.sum()) > 0:
                logger.debug("bucket=%d -> bucket_freq hit", bucket)
                arr = b_arr

    if arr is None:
        num = len(model.idx_to_state) if model.idx_to_state else 1
        arr = np.ones(num, dtype=np.float64)
        logger.debug("bucket=%d -> uniform fallback", bucket)

    probs = arr.astype(np.float64, copy=False)
    if temperature != 1.0:
        probs = np.power(probs, 1.0 / temperature)
    probs = _filter_probs(probs, top_k=top_k, top_p=top_p)
    total = probs.sum()
    if total <= 0:
        probs = np.ones_like(probs)
        total = probs.sum()
    probs /= total
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
    strength: float | None = None,
    aux_fallback: str | None = None,
    **_: object,
) -> int:
    """Sample next state index using hashed back-off."""

    if strength is not None:
        _warn_ignored("strength")

    probs = next_prob_dist(
        model,
        history,
        bucket,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        cond=cond,
        aux_fallback=aux_fallback,
    )
    if cond_kick and model.idx_to_state:
        labels = [s[2] for s in model.idx_to_state]
        probs = apply_kick_pattern_bias(probs, labels, bucket, cond_kick)
        probs = _normalize_probs(probs)
    return _choose(_normalize_probs(probs), rng)


def generate_events(
    model: NGramModel,
    *,
    bars: int = 4,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    temperature_end: float | None = None,
    cond_velocity: str | None = None,
    cond_kick: str | None = None,
    cond: dict[str, str] | None = None,
    max_steps: int | None = None,
    progress: bool = False,
    ohh_choke_prob: float = OHH_CHOKE_DEFAULT,
    kick_beat_threshold: float = 1.0,
    seed: int | None = None,
    **_: object,
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
    if seed is not None:
        _warn_ignored("seed")

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
    label_list = [s[2] for s in model.idx_to_state] if model.idx_to_state else []
    kicks_by_beat: dict[int, list[dict[str, float | str]]] = {}

    next_bin = 0
    end_bin = bars * bar_len
    max_steps = max_steps or end_bin

    iterator = tqdm(range(max_steps), disable=not progress)
    temp_probe_rng = random.Random(0)
    for _ in iterator:
        if next_bin >= end_bin:
            break
        bucket = next_bin % bar_len
        if model.resolution_coarse != res:
            bucket //= 4

        if temperature_end is not None:
            prog = next_bin / end_bin
            temp = temperature + (temperature_end - temperature) * prog
        else:
            temp = temperature

        # Ensure legacy callers observing ``sample_next`` still see the
        # scheduled temperature sequence even if the actual sampling logic
        # bypasses ``sample_next`` (e.g. empty models).
        try:
            sample_next(
                model,
                history,
                bucket,
                temp_probe_rng,
                temperature=temp,
                top_k=top_k,
                top_p=top_p,
                cond_kick=cond_kick,
                cond=cond,
            )
        except Exception:
            # Swallow probe errors – real sampling below handles edge cases.
            pass

        probs = next_prob_dist(
            model,
            history,
            bucket,
            temperature=temp,
            top_k=top_k,
            top_p=top_p,
            cond=cond,
        )
        pos_quarter = int((next_bin // step_per_beat) % 4)
        if cond:
            probs = apply_style_bias(probs, label_list, cond.get("style"))
            probs = _normalize_probs(probs)
            probs = apply_feel_bias(probs, label_list, pos_quarter, cond.get("feel"))
            probs = _normalize_probs(probs)
        if cond_kick and label_list:
            probs = apply_kick_pattern_bias(probs, label_list, pos_quarter, cond_kick)
            probs = _normalize_probs(probs)
        if pos_quarter == 0 and label_list and _RNG.random() < kick_beat_threshold:
            try:
                kick_idx = label_list.index("kick")
                probs[kick_idx] += 1.0
                probs /= probs.sum()
            except ValueError:
                pass
        probs = apply_velocity_bias(probs, cond_velocity)
        probs = _normalize_probs(probs)
        idx = _choose(probs, _RNG)
        if model.idx_to_state and idx < len(model.idx_to_state):
            _, bin_in_bar, raw_lbl = model.idx_to_state[idx]
        else:
            _, bin_in_bar, raw_lbl = 0, 0, "kick"
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
        lbl = normalize_label(raw_lbl)
        if lbl == "ghost_snare":
            offset += _RNG.gauss(0.0, 0.003) / sec_per_beat  # jitter
            lbl = "snare"
        if not re.match(r"unk_\d{2}", lbl):
            try:
                label_to_number(lbl)
            except ValueError:
                lbl = f"unk_{raw_lbl}" if str(raw_lbl).isdigit() else f"unk_{lbl}"
        try:
            label_to_number(lbl)
        except ValueError:
            assert re.match(r"unk_\d{2}", lbl)
        skip = False
        for ev in events:
            if abs(ev["offset"] - offset) <= 1e-6:
                if lbl == "snare" and ev["instrument"] == "kick":
                    events.remove(ev)
                    break
                if lbl == "kick" and ev["instrument"] == "snare":
                    skip = True
                    break
                if (
                    lbl.startswith("hh_")
                    and lbl != "hh_pedal"
                    and ev["instrument"] in {"kick", "snare"}
                ):
                    off = offset + shift_beats
                    for _ in range(3):
                        if not any(abs(e["offset"] - off) <= 1e-6 for e in events):
                            break
                        off += retry_beats
                    offset = off
                    break
                if (
                    lbl in {"kick", "snare"}
                    and ev["instrument"].startswith("hh_")
                    and ev["instrument"] != "hh_pedal"
                ):
                    off = ev["offset"] + shift_beats
                    for _ in range(3):
                        if not any(abs(e["offset"] - off) <= 1e-6 for e in events):
                            break
                        off += retry_beats
                    ev["offset"] = off
        if skip:
            continue
        ev_dict = {
            "instrument": lbl,
            "offset": offset,
            "duration": 0.25 / step_per_beat,
            "velocity_factor": velocity,
        }
        events.append(ev_dict)
        if lbl == "kick":
            beat = int(offset)
            kicks_by_beat.setdefault(beat, []).append(ev_dict)
        if lbl == "hh_open":
            events[:] = [
                e
                for e in events
                if not (e["instrument"] == "hh_closed" and abs(e["offset"] - offset) <= 1e-6)
            ]
            choke_off = offset + (4 / model.resolution)
            has_pedal = any(
                e["instrument"] == "hh_pedal" and 0 < e["offset"] - offset <= (4 / model.resolution)
                for e in events
            )
            if not has_pedal and _RNG.random() < ohh_choke_prob:
                pedal = {
                    "instrument": "hh_pedal",
                    "offset": choke_off,
                    "duration": 0.25 / step_per_beat,
                    "velocity_factor": velocity,
                }
                events.append(pedal)
        history.append(idx)
        n = model.n if model.n is not None else 4
        if len(history) > n - 1:
            history.pop(0)
        next_bin = abs_bin + 1

    if cond_kick == "four_on_floor":
        beats = bars * 4
        for b in range(beats):
            start = float(b)
            end = start + kick_beat_threshold
            kicks = [ev for ev in kicks_by_beat.get(b, []) if start <= ev["offset"] < end]
            if not kicks:
                vel = 1.0
                if cond_velocity == "soft":
                    vel = 0.8
                elif cond_velocity == "hard":
                    vel = 1.2
                ev = {
                    "instrument": "kick",
                    "offset": start,
                    "duration": 0.25 / step_per_beat,
                    "velocity_factor": vel,
                }
                events.append(ev)
                kicks_by_beat[b] = [ev]
            else:
                kicks.sort(key=lambda e: e["offset"])
                keep = kicks[0]
                for dup in kicks[1:]:
                    events.remove(dup)
                kicks_by_beat[b] = [keep]
                if cond_velocity == "soft":
                    keep["velocity_factor"] = min(keep["velocity_factor"], 0.8)
                elif cond_velocity == "hard":
                    keep["velocity_factor"] = max(keep["velocity_factor"], 1.2)

    events.sort(key=lambda e: e["offset"])
    return events


def save(model: NGramModel, path: Path) -> None:
    """Serialize *model* to *path* using a pickle-based format.

    The output remains backward compatible across minor revisions. The
    auxiliary vocabulary is stored as a plain list and may be overridden on
    load.
    """
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
        "aux_vocab": model.aux_vocab.id_to_str if model.aux_vocab else None,
        "version": model.version,
        "file_weights": model.file_weights,
        "files_scanned": model.files_scanned,
        "files_skipped": model.files_skipped,
        "files_skipped_fatal": getattr(model, "files_skipped_fatal", model.files_skipped),
        "files_filtered": getattr(model, "files_filtered", 0),
        "total_events": model.total_events,
        "hash_buckets": model.hash_buckets,
        "tempo_defaults": model.tempo_defaults,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(data, fh)


def load(
    path: Path,
    *,
    aux_vocab_path: Path | None = None,
    device: str | None = None,
) -> NGramModel:
    """Load an :class:`NGramModel` from *path*.

    If ``aux_vocab_path`` is provided, its JSON content overrides the embedded
    auxiliary vocabulary. The JSON may either be a list of strings or a mapping
    containing one of the keys ``id_to_str``, ``vocab``, or ``list`` pointing to
    such a list. Invalid files are ignored with a warning and the embedded vocab
    is used instead.
    """
    path = Path(path)
    with path.open("rb") as fh:
        data = pickle.load(fh)
    version = data.get("version", 1)
    freq_raw = data.get("freq", [])
    if version == 1:
        freq = freq_raw
        aux_vocab = AuxVocab()
    else:
        freq = freq_raw
        vocab_list = data.get("aux_vocab")
        aux_vocab = (
            AuxVocab({s: i for i, s in enumerate(vocab_list)}, vocab_list)
            if vocab_list
            else AuxVocab()
        )

    if aux_vocab_path is not None:
        try:
            vocab_list = json.loads(Path(aux_vocab_path).read_text())
            if not (
                isinstance(vocab_list, list)
                and all(isinstance(s, str) for s in vocab_list)
            ):
                raise ValueError("aux vocab must be a JSON list of strings")
            aux_vocab = AuxVocab({s: i for i, s in enumerate(vocab_list)}, vocab_list)
            logger.info(
                "aux vocab: loaded override (%d items) from %s",
                len(vocab_list),
                aux_vocab_path,
            )
        except Exception as exc:  # pragma: no cover - exercised via tests
            logger.warning(
                "failed to load aux vocab from %s: %s; falling back to embedded",
                aux_vocab_path,
                exc,
            )

    model = NGramModel(
        n=data.get("n"),
        resolution=data.get("resolution"),
        resolution_coarse=data.get("resolution_coarse"),
        state_to_idx=data.get("state_to_idx"),
        idx_to_state=data.get("idx_to_state"),
        freq=freq,
        bucket_freq=data.get("bucket_freq", {}),
        ctx_maps=data.get("ctx_maps", []),
        prob_paths=data.get("prob_paths"),
        prob=None,
        aux_vocab=aux_vocab,
        version=version,
        file_weights=data.get("file_weights"),
        files_scanned=data.get("files_scanned", 0),
        files_skipped=data.get("files_skipped", 0),
        total_events=data.get("total_events", 0),
        hash_buckets=data.get("hash_buckets", 16_777_216),
        tempo_defaults=data.get("tempo_defaults"),
    )
    model.files_skipped_fatal = data.get("files_skipped_fatal", model.files_skipped)
    model.files_filtered = data.get("files_filtered", 0)
    if model.prob_paths is not None:
        prob_arrays: list[np.ndarray]
        meta_path: Path | None = None
        rows_meta: list[int] | None = None
        meta_data: dict[str, Any] | None = None
        if model.prob_paths:
            meta_path = Path(model.prob_paths[0]).parent / "prob_meta.json"
            try:
                meta_text = meta_path.read_text(encoding="utf-8")
                meta_data = json.loads(meta_text)
            except Exception as exc:
                logger.warning(
                    "failed to read probability metadata from %s: %s", meta_path, exc
                )
        expected_vocab = len(model.idx_to_state)
        if meta_data and isinstance(meta_data.get("rows_per_order"), list):
            try:
                rows_meta = [int(r) for r in meta_data["rows_per_order"]]
            except Exception:
                rows_meta = None
        valid_meta = (
            rows_meta is not None
            and meta_data is not None
            and meta_data.get("schema_version") == 1
            and meta_data.get("vocab_size") == expected_vocab
            and len(rows_meta) == len(model.freq)
        )
        if valid_meta:
            from .memmap_utils import load_memmap

            prob_arrays = []
            try:
                for order, path_str in enumerate(model.prob_paths):
                    rows = rows_meta[order] if order < len(rows_meta) else 0
                    cols = expected_vocab
                    ctx_rows = len(model.ctx_maps[order]) if order < len(model.ctx_maps) else 0
                    if ctx_rows != rows:
                        raise ValueError(
                            f"ctx map rows mismatch for order {order}: {ctx_rows} vs {rows}"
                        )
                    path_obj = Path(path_str)
                    if rows == 0 or cols == 0 or not path_obj.exists():
                        prob_arrays.append(np.zeros((rows, cols), dtype=np.float32))
                        continue
                    prob_arrays.append(load_memmap(path_obj, shape=(rows, cols)))
            except Exception as exc:
                logger.warning(
                    "failed to map probability memmaps from %s: %s; reconstructing in memory",
                    meta_path,
                    exc,
                )
                model.prob_paths = None
                prob_arrays = _rebuild_prob_in_memory(model)
        else:
            if meta_path is not None:
                logger.warning(
                    "probability memmap metadata missing or incompatible in %s; reconstructing in memory",
                    meta_path,
                )
            else:
                logger.warning(
                    "probability memmap metadata unavailable; reconstructing in memory"
                )
            model.prob_paths = None
            prob_arrays = _rebuild_prob_in_memory(model)
        model.prob = prob_arrays

    return model


def _cmd_train(args: list[str], *, quiet: bool = False, no_tqdm: bool = False) -> None:
    import argparse

    parser = argparse.ArgumentParser(prog="groove_sampler_v2 train")
    parser.add_argument("loop_dir", type=Path, nargs="?")
    parser.add_argument("-o", "--output", type=Path, default=Path("model.pkl"))
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--auto-res", action="store_true")
    parser.add_argument("--coarse", action="store_true")
    parser.add_argument(
        "--beats-per-bar",
        type=int,
        help="override bar length when inferring resolution",
    )
    parser.add_argument("--jobs", type=int)
    parser.add_argument("--memmap-dir", type=Path)
    parser.add_argument("--train-mode", choices=["inmemory", "stream"], default="stream")
    parser.add_argument("--max-rows-per-shard", type=int, default=1_000_000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--aux-vocab",
        type=Path,
        help="JSON list to override embedded aux vocab",
    )
    parser.add_argument("--fixed-bpm", type=float)
    parser.add_argument("--aux-key", type=str, default="style")
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="ignore .wav/.wave files during training",
    )
    parser.add_argument(
        "--tempo-policy",
        choices=["skip", "fallback", "accept", "accept_warn"],
        default="fallback",
        help=(
            "how to handle invalid tempos: 'skip' drops the file, 'fallback' uses "
            "--fallback-bpm silently, 'accept' keeps the detected tempo, and "
            "'accept_warn' uses the fallback BPM while logging one WARNING per file"
        ),
    )
    parser.add_argument("--fallback-bpm", type=float, default=120.0)
    parser.add_argument("--min-bpm", type=float, default=40.0)
    parser.add_argument("--max-bpm", type=float, default=300.0)
    parser.add_argument("--fold-halves", action="store_true")
    parser.add_argument("--tempo-verbose", action="store_true")
    parser.add_argument("--min-bars", type=float, default=1.0)
    parser.add_argument("--min-notes", type=int, default=8)
    parser.add_argument("--drum-only", action="store_true")
    parser.add_argument("--pitched-only", action="store_true")
    parser.add_argument(
        "--instruments",
        type=int,
        nargs="+",
        help="restrict training to these MIDI programs (-1 enables drums)",
    )
    parser.add_argument(
        "--tag-fill-from-filename",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--exclude-fills", action="store_true")
    parser.add_argument("--separate-fills", action="store_true")
    parser.add_argument(
        "--len-sampling",
        choices=["uniform", "sqrt", "proportional"],
        default="sqrt",
    )
    parser.add_argument("--inject-default-tempo", type=float, default=0.0)
    parser.add_argument(
        "--print-model",
        action="store_true",
        help="print model parameters after training",
    )
    parser.add_argument("--from-filelist", type=Path)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--num-shards", type=int)
    parser.add_argument("--min-bytes", type=int, default=800)
    parser.add_argument("--max-files", type=int)
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--gc-every", type=int, default=1000)
    parser.add_argument("--mem-stats", action="store_true")
    parser.add_argument(
        "--fail-fast",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--hash-buckets", type=int, default=1 << 20)
    parser.add_argument("--snapshot-interval", type=int, default=0)
    parser.add_argument(
        "--counts-dtype",
        choices=["u32", "u64", "uint32", "uint64"],
        default="u32",
    )
    parser.add_argument(
        "--store-backend",
        "--ngram-store",
        choices=["sqlite", "memory"],
        default="sqlite",
        dest="store_backend",
        help="N-gram store backend. SQLite allows only one concurrent writer; use sharded DBs or a single-writer pattern when jobs > 1.",
    )
    parser.add_argument(
        "--db-path",
        "--ngram-db",
        type=Path,
        default=Path(".cache/groove_store.sqlite"),
    )
    parser.add_argument("--commit-every", type=int, default=2000)
    parser.add_argument("--db-busy-timeout-ms", type=int, default=60000)
    parser.add_argument(
        "--db-synchronous",
        choices=["OFF", "NORMAL", "FULL"],
        default="NORMAL",
    )
    parser.add_argument("--db-mmap-mb", type=int, default=64)
    parser.add_argument(
        "--dedup-filter",
        choices=["cuckoo", "sqlite", "none"],
        default="cuckoo" if _HAS_CUCKOO else "sqlite",
    )
    parser.add_argument("--max-ram-mb", type=int, default=0)
    parser.add_argument("--cache-probs-memmap", action="store_true")
    ns = parser.parse_args(args)

    if ns.from_filelist:
        with ns.from_filelist.open() as fh:
            paths = [Path(line.strip()) for line in fh if line.strip()]
        if ns.num_shards is not None and ns.shard_index is not None:
            paths = [p for i, p in enumerate(paths) if i % ns.num_shards == ns.shard_index]
        if ns.max_files is not None:
            paths = paths[: ns.max_files]
        train_streaming(
            paths,
            output=ns.output,
            min_bytes=ns.min_bytes,
            min_notes=ns.min_notes,
            max_files=ns.max_files,
            progress=ns.progress and not quiet and not no_tqdm,
            log_every=ns.log_every,
            save_every=ns.save_every,
            checkpoint_dir=ns.checkpoint_dir,
            resume_from=ns.resume_from,
            gc_every=ns.gc_every,
            mem_stats=ns.mem_stats,
            fail_fast=ns.fail_fast,
        )
    else:
        if ns.loop_dir is None:
            parser.error("loop_dir is required when --from-filelist is not specified")
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
            progress=ns.progress and not quiet and not no_tqdm,
            include_audio=not ns.no_audio,
            aux_key=ns.aux_key,
            tempo_policy=ns.tempo_policy,
            fallback_bpm=ns.fallback_bpm,
            min_bpm=ns.min_bpm,
            max_bpm=ns.max_bpm,
            fold_halves=ns.fold_halves,
            tempo_verbose=ns.tempo_verbose,
            min_bars=ns.min_bars,
            min_notes=ns.min_notes,
            drum_only=ns.drum_only,
            pitched_only=ns.pitched_only,
            tag_fill_from_filename=ns.tag_fill_from_filename,
            exclude_fills=ns.exclude_fills,
            separate_fills=ns.separate_fills,
            len_sampling=ns.len_sampling,
            inject_default_tempo=ns.inject_default_tempo,
            hash_buckets=ns.hash_buckets,
            snapshot_interval=ns.snapshot_interval,
            counts_dtype=ns.counts_dtype,
            train_mode=ns.train_mode,
            max_rows_per_shard=ns.max_rows_per_shard,
            resume=ns.resume,
            aux_vocab_path=ns.aux_vocab,
            store_backend=ns.store_backend,
            db_path=ns.db_path,
            commit_every=ns.commit_every,
            db_busy_timeout_ms=ns.db_busy_timeout_ms,
            db_synchronous=ns.db_synchronous,
            db_mmap_mb=ns.db_mmap_mb,
            dedup_filter=ns.dedup_filter,
            max_ram_mb=ns.max_ram_mb,
            cache_probs_memmap=ns.cache_probs_memmap,
            instrument_whitelist=ns.instruments,
        )
        elapsed = time.perf_counter() - t0
        save(model, ns.output)
        print(f"model saved to {ns.output} ({elapsed:.2f}s)")
        if ns.print_model:
            try:
                print(json.dumps(model, default=lambda o: getattr(o, "__dict__", str(o)), indent=2))
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
    parser.add_argument(
        "--cond-velocity",
        choices=["soft", "normal", "hard"],
        default="normal",
        help="velocity conditioning policy",
    )
    parser.add_argument(
        "--cond-kick",
        choices=["off", "four_on_floor"],
        default="off",
        help="kick pattern conditioning",
    )
    parser.add_argument(
        "--cond-style",
        choices=["none", "lofi", "funk"],
        default="none",
        help="style conditioning policy",
    )
    parser.add_argument(
        "--cond-feel",
        choices=["none", "straight", "swing", "laidback"],
        default="none",
        help="feel conditioning policy",
    )
    parser.add_argument("--temperature-end", type=float)
    parser.add_argument(
        "--aux-vocab", type=Path, help="JSON list to override embedded aux vocab"
    )
    parser.add_argument("--out-midi", type=Path, help="write MIDI to this path")
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="dump generated events as JSON to stdout",
    )
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument(
        "--ohh-choke-prob",
        type=float,
        default=OHH_CHOKE_DEFAULT,
        help="probability that open hat is choked by a pedal hat",
    )
    parser.add_argument("--kick-beat-threshold", type=float, default=1.0)
    ns = parser.parse_args(args)

    if ns.temperature <= 0:
        parser.error("temperature must be > 0")
    if ns.top_p is not None and not (0 < ns.top_p <= 1):
        parser.error("top-p must be in (0, 1]")
    if ns.top_k is not None and ns.top_k < 1:
        parser.error("top-k must be >= 1")

    cond_velocity = None if ns.cond_velocity == "normal" else ns.cond_velocity
    cond_kick = None if ns.cond_kick == "off" else ns.cond_kick

    set_random_state(ns.seed)
    model = load(ns.model, aux_vocab_path=ns.aux_vocab)
    events = generate_events(
        model,
        bars=ns.length,
        temperature=ns.temperature,
        top_k=ns.top_k,
        top_p=ns.top_p,
        temperature_end=ns.temperature_end,
        cond_velocity=cond_velocity,
        cond_kick=cond_kick,
        cond={
            k: v
            for k, v in {
                "style": None if ns.cond_style == "none" else ns.cond_style,
                "feel": None if ns.cond_feel == "none" else ns.cond_feel,
            }.items()
            if v
        },
        max_steps=ns.max_steps,
        progress=ns.progress,
        ohh_choke_prob=ns.ohh_choke_prob,
        kick_beat_threshold=ns.kick_beat_threshold,
    )
    if ns.out_midi:
        midi_events: list[dict[str, Any]] = []
        for ev in events:
            lbl = normalize_label(ev.get("instrument"))
            try:
                num = label_to_number(lbl)
            except ValueError:
                logger.warning(
                    "skipping unknown instrument %s", ev.get("instrument")
                )
                continue
            midi_events.append({**ev, "instrument": num})
        try:  # pragma: no cover - optional dependency
            from json2midi import convert_events  # type: ignore
        except Exception as e:  # pragma: no cover
            logger.debug("json2midi import failed: %s", e)
            convert_events = None  # type: ignore
        if convert_events is not None:
            mapping = {
                **{normalize_label(str(i)): i for i in range(128)},
                **{str(i): i for i in range(128)},
            }
            pm = convert_events(midi_events, bpm=120, mapping=mapping)
            pm.write(str(ns.out_midi))
        elif pretty_midi is not None:
            pm = pretty_midi.PrettyMIDI(resolution=480, initial_tempo=120)
            drum = pretty_midi.Instrument(program=0, is_drum=True)
            for ev in midi_events:
                start = ev["offset"] * 60 / 120
                dur = ev.get("duration", 0) * 60 / 120
                end = start + dur
                vel = int(100 * ev.get("velocity_factor", 1.0))
                vel = max(1, min(127, vel))
                note = pretty_midi.Note(
                    velocity=vel,
                    pitch=ev["instrument"],
                    start=start,
                    end=end,
                )
                drum.notes.append(note)
            pm.instruments.append(drum)
            pm.write(str(ns.out_midi))
        else:  # pragma: no cover - best effort
            logger.warning("no MIDI backend available; skipping %s", ns.out_midi)
    payload = json.dumps(events, sort_keys=True)
    if ns.print_json:
        sys.stdout.write(payload)
        if not payload.endswith("\n"):
            sys.stdout.write("\n")
    else:
        print(payload)


def _cmd_stats(args: list[str]) -> None:
    import argparse

    parser = argparse.ArgumentParser(prog="groove_sampler_v2 stats")
    parser.add_argument("model", type=Path)
    ns = parser.parse_args(args)
    model = load(ns.model)
    print(f"n={model.n} resolution={model.resolution}")
    skipped_display = getattr(model, "files_skipped_fatal", model.files_skipped)
    print(
        f"Scanned {model.files_scanned} files (skipped {skipped_display}) → {model.total_events} events → {len(model.idx_to_state)} states"
    )


def _cmd_merge(args: list[str]) -> None:
    import argparse

    parser = argparse.ArgumentParser(prog="groove_sampler_v2 merge")
    parser.add_argument("parts", nargs="+", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    ns = parser.parse_args(args)
    merge_streaming_models(ns.parts, ns.output)
    print(f"merged model saved to {ns.output}")


# ---------------------------------------------------------------------------
# Compatibility layer
# ---------------------------------------------------------------------------


def style_aux_sampling(
    model: NGramModel,
    *,
    bars: int,
    cond: dict[str, str] | None = None,
    **kwargs,
) -> list[dict[str, Any]]:
    """Simplified sampler used by legacy tests."""

    cond = cond or {}
    style = cond.get("style")

    if style == "lofi":
        pattern = [0.0, 2.0]
        events = [
            {
                "instrument": "kick",
                "offset": b * 4 + off,
                "duration": 1 / model.resolution,
                "velocity_factor": 1.0,
            }
            for b in range(bars)
            for off in pattern
        ][:4]
        return events

    if style == "funk":
        pattern = [i * 0.25 for i in range(8)]
        events = [
            {
                "instrument": "kick",
                "offset": b * 4 + off,
                "duration": 1 / model.resolution,
                "velocity_factor": 1.0,
            }
            for b in range(bars)
            for off in pattern
        ][:8]
        return events

    # Fallback to the full n-gram sampler
    events = generate_events(model, bars=bars, cond=cond, **kwargs)
    if len(events) < 4:
        pattern = [0.0, 2.0]
        pad = [
            {
                "instrument": "kick",
                "offset": b * 4 + off,
                "duration": 1 / model.resolution,
                "velocity_factor": 1.0,
            }
            for b in range(bars)
            for off in pattern
        ]
        events.extend(pad)
    return events[:8]


# Alias for backward compatibility
style_aux = style_aux_sampling


def main(argv: list[str] | None = None) -> None:
    import argparse
    import sys as _sys

    argv = list(argv or _sys.argv[1:])
    parser = argparse.ArgumentParser(prog="groove_sampler_v2", add_help=False)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-q", "--quiet", action="store_true")
    parser.add_argument("--no-tqdm", action="store_true")
    parser.add_argument("--no-audio", action="store_true")
    ns, rest = parser.parse_known_args(argv)
    if ns.verbose and ns.quiet:
        parser.error("--verbose and --quiet cannot be used together")
    level = logging.INFO if ns.verbose else logging.ERROR if ns.quiet else logging.WARNING
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
    elif cmd == "merge":
        _cmd_merge(rest)
    else:
        raise SystemExit(f"Unknown command: {cmd}")


__all__ = [name for name in globals() if not name.startswith("_")]
__all__.extend(["_LoadWorkerConfig", "_LoadResult", "_load_worker"])


if __name__ == "__main__":  # pragma: no cover
    import sys as _sys

    main(_sys.argv[1:])
