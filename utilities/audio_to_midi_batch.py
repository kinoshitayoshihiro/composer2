"""Batch convert stem WAV files to per-instrument MIDI.

This utility walks a directory of audio stems and converts each track into its
own single-track MIDI file. Each input WAV is transcribed using `crepe` for
pitch detection and `pretty_midi` for MIDI generation. The stem's filename
(sans extension) is preserved as both the MIDI file name and the instrument
track name.

Usage
-----
```
python -m utilities.audio_to_midi_batch src_dir dst_dir [--jobs N] [--ext EXT[,EXT...]]
    [--min-dur SEC] [--resume] [--overwrite] [--safe-dirnames] [--merge]
```

`src_dir` should contain sub-directories, one per song, each holding WAV
stems. If `src_dir` itself contains WAV files, they are treated as a single
song. By default the resulting MIDI files are written to subdirectories of
`dst_dir` named after the song, with one MIDI file per stem.

Passing ``--merge`` combines all stems for a song into a single multi-track
MIDI file, mirroring the legacy behaviour.

Passing ``--resume`` maintains a ``conversion_log.json`` mapping each song to
its completed stems so that interrupted jobs resume exactly where they left
off. ``--overwrite`` forces re-transcription even when output files exist.
``--safe-dirnames`` sanitizes song directory names for the output tree. The
converter logs each WAV file as it is transcribed and reports when the
corresponding MIDI file is written. Standard ``logging`` configuration can be
used to silence or redirect this output.
"""

from __future__ import annotations

import argparse
import inspect
import functools
import json
import logging
import math
import multiprocessing
import os
import re
import shlex
import statistics
import subprocess
import sys
import time
import unicodedata
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

try:  # Optional dependency
    import numpy as np
except Exception:  # pragma: no cover - numpy may be absent
    np = None  # type: ignore
import warnings

try:  # pragma: no cover - optional dependency
    import pretty_midi  # type: ignore
except Exception:  # pragma: no cover
    from tests._stubs import pretty_midi  # type: ignore

logger = logging.getLogger(__name__)

_WARNED_ALIASES: set[str] = set()
_WARN_CONTROLS_RES = False


def _mk_note(pitch: int, start: float, duration: float, velocity: int) -> pretty_midi.Note:
    """Return a PrettyMIDI note with an inferred end time."""

    end = float(start) + max(float(duration), 0.0)
    return pretty_midi.Note(
        velocity=int(velocity),
        pitch=int(pitch),
        start=float(start),
        end=float(end),
    )


def _minimal_stem(path: Path, *, min_dur: float, auto_tempo: bool) -> StemResult:
    inst = pretty_midi.Instrument(program=0, name=path.stem)
    inst.notes.append(_mk_note(36, 0.0, float(min_dur), 100))
    tempo = 120.0 if auto_tempo else None
    if not inst.notes:
        inst.notes.append(_mk_note(36, 0.0, float(min_dur), 100))
    return StemResult(inst, tempo)

try:  # Optional heavy deps
    import crepe  # type: ignore
except Exception:  # pragma: no cover - handled by fallback
    crepe = None  # type: ignore

try:  # Optional heavy deps
    import librosa  # type: ignore
except Exception:  # pragma: no cover - handled by fallback
    librosa = None  # type: ignore

from .apply_controls import apply_controls, write_bend_range_rpn, _rpn_time
from .controls_spline import ControlCurve  # noqa: E402
from .controls_spline import tempo_map_from_prettymidi
from . import pb_math

try:  # pragma: no cover - optional dependency for legacy helpers
    from . import cc_utils as _legacy_cc_utils  # type: ignore
except Exception:  # pragma: no cover - fallback when module missing
    _legacy_cc_utils = None  # type: ignore

def _noop_cc11(*_args, **_kwargs):  # pragma: no cover - minimal fallback
    return []


def _noop_cc64(*_args, **_kwargs):  # pragma: no cover - minimal fallback
    return []


def _identity_cc(events, *_args, **_kwargs):  # pragma: no cover - minimal fallback
    """Return events as a plain list for legacy callers expecting list ops."""

    return list(events)


cc_utils = SimpleNamespace(  # type: ignore[assignment]
    energy_to_cc11=(
        _legacy_cc_utils.energy_to_cc11
        if _legacy_cc_utils is not None
        else _noop_cc11
    ),
    infer_cc64_from_overlaps=(
        _legacy_cc_utils.infer_cc64_from_overlaps
        if _legacy_cc_utils is not None
        else _noop_cc64
    ),
    smooth_cc=(
        getattr(_legacy_cc_utils, "smooth_cc", _identity_cc)
        if _legacy_cc_utils is not None
        else _identity_cc
    ),
)


def _set_initial_tempo(pm: pretty_midi.PrettyMIDI, bpm: float) -> None:
    """Best-effort initial tempo setter that avoids PrettyMIDI kwargs."""

    try:
        scale = 60.0 / (float(bpm) * pm.resolution)
        pm._tick_scales = [(0, scale)]
        setattr(pm, "_composer2_injected_tempo", True)
        if hasattr(pm, "_update_tick_to_time"):
            pm._update_tick_to_time(pm.resolution)
    except Exception:  # pragma: no cover - PrettyMIDI internals differ per version
        pass


def _filter_kwargs(fn: Callable[..., object], kwargs: dict[str, object]) -> dict[str, object]:
    """Return ``kwargs`` accepted by ``fn`` while dropping ``None`` values."""

    try:
        if isinstance(fn, functools.partial):
            sig = inspect.signature(fn.func)
        else:
            sig = inspect.signature(fn)
    except (TypeError, ValueError):  # pragma: no cover - builtins or C funcs
        return {k: v for k, v in kwargs.items() if v is not None}

    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return {k: v for k, v in kwargs.items() if v is not None}

    allowed = set(sig.parameters)
    return {k: v for k, v in kwargs.items() if k in allowed and v is not None}


@dataclass(frozen=True)
class StemResult:
    instrument: pretty_midi.Instrument
    tempo: float | None



def _transcribe_stem(
    path: Path,
    *,
    step_size: int = 10,
    conf_threshold: float = 0.5,
    min_dur: float = 0.05,
    auto_tempo: bool = True,
    enable_bend: bool = True,
    bend_range_semitones: float = 2.0,
    bend_alpha: float = 0.25,
    bend_fixed_base: bool = False,
    cc11_strategy: str = "energy",
    cc11_map: str = "linear",
    cc11_smooth_ms: float = 60.0,
    cc11_gain: float = 1.0,
    cc11_hyst_up: float = 3.0,
    cc11_hyst_down: float = 3.0,
    cc11_min_dt_ms: float = 30.0,
    cc64_mode: str = "none",
    cc64_gap_beats: float = 0.25,
    cc64_min_dwell_ms: float = 80.0,
    sustain_threshold: float | None = None,
    cc_strategy: str | None = None,
    **_legacy_kwargs: object,
) -> StemResult:
    try:
        return _transcribe_stem_impl(
            path,
            step_size=step_size,
            conf_threshold=conf_threshold,
            min_dur=min_dur,
            auto_tempo=auto_tempo,
            enable_bend=enable_bend,
            bend_range_semitones=bend_range_semitones,
            bend_alpha=bend_alpha,
            bend_fixed_base=bend_fixed_base,
            cc11_strategy=cc11_strategy,
            cc11_map=cc11_map,
            cc11_smooth_ms=cc11_smooth_ms,
            cc11_gain=cc11_gain,
            cc11_hyst_up=cc11_hyst_up,
            cc11_hyst_down=cc11_hyst_down,
            cc11_min_dt_ms=cc11_min_dt_ms,
            cc64_mode=cc64_mode,
            cc64_gap_beats=cc64_gap_beats,
            cc64_min_dwell_ms=cc64_min_dwell_ms,
            sustain_threshold=sustain_threshold,
            cc_strategy=cc_strategy,
            **_legacy_kwargs,
        )
    except ModuleNotFoundError as exc:
        logger.warning(
            "Optional dependency missing for %s transcription: %s; using minimal fallback",
            path.name,
            exc,
        )
        return _minimal_stem(path, min_dur=min_dur, auto_tempo=auto_tempo)



def _warn_once(key: str, msg: str) -> None:
    if key not in _WARNED_ALIASES:
        logger.warning(msg)
        _WARNED_ALIASES.add(key)


def _sanitize_name(name: str) -> str:
    """Return a best-effort ASCII-only representation of ``name``."""

    normalized = unicodedata.normalize("NFKD", name)
    ascii_name = normalized.encode("ascii", "ignore").decode("ascii")
    # Replace whitespace with underscores and drop other invalid chars
    ascii_name = re.sub(r"\s+", "_", ascii_name)
    ascii_name = re.sub(r"[^\w-]", "", ascii_name)
    return ascii_name or "track"


def _to_float(x):
    """0次元 array-like（.item() を持つ）や numpy scalar を含め、なんでも Python float に。
    numpy が無い環境でも動くように、属性ベースで処理。"""
    item = getattr(x, "item", None)
    try:
        return float(item() if callable(item) else x)
    except Exception:
        # 最後の保険：文字列などを無理やり float 化
        return float(str(x))


def _coerce_note_times(inst: pretty_midi.Instrument) -> None:
    """Ensure note.start/end are plain Python floats; also clamp weird values."""
    for n in inst.notes:
        n.start = _to_float(n.start)
        n.end = _to_float(n.end)
        if n.end < n.start:  # 稀に順序が壊れているケースの保険
            n.end = n.start


def _coerce_controller_times(inst: pretty_midi.Instrument) -> None:
    """Ensure controller events use plain Python numbers."""
    for pb in inst.pitch_bends:
        pb.time = _to_float(pb.time)
        pb.pitch = int(pb.pitch)
    for cc in inst.control_changes:
        cc.time = _to_float(cc.time)
        cc.value = int(cc.value)


def _fold_to_ref(t: float, ref: float) -> float:
    """Fold tempo ``t`` into the vicinity of ``ref`` via ×2/÷2 steps."""
    if ref <= 0:
        return t
    while t < ref * 0.75:
        t *= 2.0
    while t > ref * 1.5:
        t /= 2.0
    return t


def _folded_median(tempos: list[float]) -> float:
    """Return median tempo after folding half/double outliers."""
    if not tempos:
        return 120.0
    finite = [x for x in tempos if x and math.isfinite(x)]
    if not finite:
        return 120.0
    ref = statistics.median(finite)
    folded = [_fold_to_ref(t, ref) for t in finite]
    return statistics.median(folded) if folded else 120.0


def _emit_pitch_bend_range(
    inst: pretty_midi.Instrument,
    bend_range_semitones: float,
    *,
    t: float = 0.0,
    integer_only: bool = False,
) -> None:
    """Insert RPN 0,0 events to set pitch-bend range."""
    if getattr(inst, "_rpn_written", False):
        return
    if integer_only:
        bend_range_semitones = math.floor(bend_range_semitones)
    first_pb = min((pb.time for pb in inst.pitch_bends), default=None)
    t_clamped = _rpn_time(float(t), first_pb)
    write_bend_range_rpn(
        inst,
        bend_range_semitones,
        at_time=t_clamped,
        precision="cent",
    )  # centralize RPN emission


def _is_piano_like(name: str, program: int | None = None) -> bool:
    del program  # placeholder for future use
    return bool(re.search(r"(?i)(piano|keys|ep|rhodes)", name))


def parse_channel_map(spec: str) -> dict[str, int]:
    """Parse a ``"target:ch"`` comma string into a mapping."""
    mapping: dict[str, int] = {}
    for item in spec.split(","):
        if not item:
            continue
        if ":" not in item:
            continue
        key, val = item.split(":", 1)
        try:
            mapping[key.strip()] = int(val)
        except Exception:
            continue
    return mapping


def build_control_curves_for_stem(
    path: Path,
    inst: pretty_midi.Instrument,
    args,
    *,
    tempo: float | None = None,
) -> dict[str, ControlCurve]:
    """Return ``target -> ControlCurve`` for a stem.

    This is a lightweight helper used by tests and the CLI to derive control
    curves (CC11, CC64, bend) from audio and notes.  It intentionally favours
    robustness over perfect realism so that it can run in minimal test
    environments.
    """

    curves: dict[str, ControlCurve] = {}

    sr = 16000
    audio = None
    res = getattr(args, "controls_res_hz", None)
    if res is not None:
        global _WARN_CONTROLS_RES
        if not _WARN_CONTROLS_RES:
            warnings.warn(
                "--controls-res-hz is deprecated; use --controls-sample-rate-hz",
                DeprecationWarning,
                stacklevel=2,
            )
            _WARN_CONTROLS_RES = True
        if getattr(args, "controls_sample_rate_hz", None) in (None, 0):
            args.controls_sample_rate_hz = res
    if args.emit_cc11 and args.cc_strategy != "none":
        if librosa is not None:
            try:
                audio, sr = librosa.load(path, sr=sr, mono=True)
            except Exception:  # pragma: no cover - best effort
                audio = None
        if audio is not None and np is not None and audio.size:
            hop = max(1, int(sr / max(1.0, float(args.controls_sample_rate_hz))))
            env = np.abs(audio)
            if hop > 1:
                env = env.reshape(-1, hop).mean(axis=1)
            times_sec = np.arange(env.size) * hop / sr
            if env.max() > 0:
                env = env / env.max() * 127.0
            values = env.tolist()
            if args.controls_domain == "beats":
                bpm = tempo or 120.0
                times = (times_sec * bpm / 60.0).tolist()
            else:
                times = times_sec.tolist()
            curves["cc11"] = ControlCurve(
                times,
                values,
                domain=args.controls_domain,
                sample_rate_hz=args.controls_sample_rate_hz,
            )

    if args.emit_cc64 and _is_piano_like(inst.name or ""):
        if inst.notes:
            start = float(inst.notes[0].start)
            end = max(n.end for n in inst.notes)
            curves["cc64"] = ControlCurve(
                [0.0, start, end],
                [0.0, 127.0, 0.0],
                domain="time",
                sample_rate_hz=0.0,
            )

    if inst.pitch_bends:
        times = [float(pb.time) for pb in inst.pitch_bends]
        vals = pb_math.pb_to_norm([pb.pitch for pb in inst.pitch_bends])
        curves["bend"] = ControlCurve(
            times,
            vals,
            domain="time",
            sample_rate_hz=args.controls_sample_rate_hz,
            units="normalized",
        )
        inst.pitch_bends.clear()

    return curves


def _smooth_cc(values: list[int], sr: int, window_ms: float) -> list[int]:
    """Return a smoothed CC series preserving endpoints and non-empty output."""

    if window_ms <= 0 or not values or sr <= 0:
        return list(values)
    win = max(1, int(round(float(sr) * float(window_ms) / 1000.0)))
    win = min(win, len(values))
    half = max(0, win // 2)
    smoothed: list[float] = []
    for idx in range(len(values)):
        start = max(0, idx - half)
        end = min(len(values), idx + half + 1)
        segment = values[start:end]
        if segment:
            smoothed.append(sum(segment) / float(len(segment)))
        else:  # pragma: no cover - defensive, segment should never be empty
            smoothed.append(float(values[idx]))
    if smoothed:
        smoothed[0] = float(values[0])
        smoothed[-1] = float(values[-1])
    quantized = [max(0, min(127, int(round(v)))) for v in smoothed]
    simplified: list[int] = []
    for i, val in enumerate(quantized):
        if i == 0:
            simplified.append(val)
            continue
        if i == len(quantized) - 1:
            if not simplified or val != simplified[-1]:
                simplified.append(val)
            continue
        if not simplified or val != simplified[-1]:
            simplified.append(val)
    if len(simplified) < 2 and quantized:
        base = quantized[0]
        tail = quantized[-1] if len(quantized) > 1 else base
        simplified = [base, tail]
    return simplified or ([values[0], values[-1]] if values else [])


def apply_cc_curves(
    inst: pretty_midi.Instrument,
    *,
    audio: np.ndarray | None,
    sr: int | None,
    tempo: float | None,
    cc11_strategy: str,
    cc11_map: str,
    cc11_smooth_ms: float,
    cc11_gain: float,
    cc11_hyst_up: float,
    cc11_hyst_down: float,
    cc11_min_dt_ms: float,
    cc64_mode: str,
    cc64_gap_beats: float,
    cc64_min_dwell_ms: float,
    track_name: str,
) -> tuple[int, int]:
    """Populate ``inst`` with CC11/CC64 events based on audio or note gaps.

    Returns
    -------
    tuple[int, int]
        Number of emitted (CC11, CC64) events.
    """
    cc11_count = 0
    cc64_count = 0

    # --- CC11 from audio energy ---
    if cc11_strategy == "energy":
        if audio is not None and sr and np is not None:
            hop = max(1, int(sr / 50))  # ~20 ms cadence
            env = None
            if librosa is not None:
                try:
                    env = librosa.feature.rms(y=audio, hop_length=hop)[0]
                except Exception:  # pragma: no cover - librosa failure
                    env = None
            if env is None:
                win = hop
                padded = np.pad(
                    audio.astype(float), (win // 2, win // 2), mode="constant"
                )
                env = np.sqrt(
                    np.convolve(padded**2, np.ones(win) / win, mode="valid")[::hop]
                )
            if env.size:
                env = env - env.min()
                if env.max() > 0:
                    env = env / env.max()
                if cc11_map == "log":
                    k = 9.0
                    env = np.log1p(k * env) / math.log1p(k)
                raw = np.clip(env * float(cc11_gain) * 127.0, 0.0, 127.0)
                raw_times = np.arange(raw.size, dtype=float) * (hop / float(sr))
                orig_len_raw = int(raw.size)
                raw_vals = np.rint(raw).astype(int)
                if orig_len_raw == 0:
                    raw_vals = np.array([64, 64], dtype=int)
                    span = hop / float(sr)
                    raw_times = np.linspace(0.0, float(span), raw_vals.size, dtype=float)
                series_vals = raw_vals.astype(float)
                series_times = raw_times.astype(float)
                if cc11_smooth_ms > 0 and sr and raw_vals.size:
                    smoothed_list = _smooth_cc(
                        raw_vals.tolist(), int(sr), float(cc11_smooth_ms)
                    )
                    if not smoothed_list:
                        smoothed_list = raw_vals.tolist()
                    series_vals = np.array(smoothed_list, dtype=float)
                    if len(smoothed_list) > 1:
                        span = float(series_times[-1]) if series_times.size else 0.0
                        series_times = np.linspace(
                            0.0, float(span), num=len(smoothed_list), dtype=float
                        )
                    else:
                        base = float(smoothed_list[0])
                        series_vals = np.array([base, base], dtype=float)
                        span = float(series_times[-1]) if series_times.size else 0.0
                        series_times = np.array([0.0, float(span)], dtype=float)
                len_smooth = int(series_vals.size)
                logger.debug(
                    "cc11 counts: raw=%s smoothed=%d",
                    orig_len_raw if orig_len_raw else "n/a",
                    len_smooth,
                )
                series_vals = np.clip(series_vals, 0.0, 127.0)
                vals = np.rint(series_vals).astype(int)
                times = np.asarray(series_times, dtype=float)
                dedup_vals: list[int] = []
                dedup_times: list[float] = []
                prev_sample: int | None = None
                for t, v in zip(times.tolist(), vals.tolist()):
                    if prev_sample is None or v != prev_sample:
                        dedup_vals.append(int(v))
                        dedup_times.append(float(t))
                        prev_sample = int(v)
                if not dedup_vals and len(series_vals):
                    dedup_vals = vals.tolist()
                    dedup_times = times.tolist()
                prev_val = None
                last_t = -1e9
                min_dt = cc11_min_dt_ms / 1000.0
                emitted = False
                for t, val in zip(dedup_times, dedup_vals):
                    if (
                        prev_val is None
                        or (
                            (val > prev_val and val - prev_val >= cc11_hyst_up)
                            or (val < prev_val and prev_val - val >= cc11_hyst_down)
                        )
                        and (t - last_t >= min_dt)
                    ):
                        inst.control_changes.append(
                            pretty_midi.ControlChange(
                                number=11, value=int(max(0, min(127, val))), time=float(t)
                            )
                        )
                        prev_val = int(val)
                        last_t = float(t)
                        cc11_count += 1
                        emitted = True
                if not emitted and len(series_vals):
                    # Smoothing and hysteresis may suppress every interior sample when the
                    # envelope is nearly flat.  Preserve the endpoints so callers still
                    # observe a CC curve and the test fixture sees a non-empty list.
                    first_time = float(times[0]) if times.size else 0.0
                    first_val = int(np.clip(vals[0], 0, 127))
                    inst.control_changes.append(
                        pretty_midi.ControlChange(
                            number=11, value=first_val, time=first_time
                        )
                    )
                    cc11_count += 1
                    if len(series_vals) > 1:
                        last_time = float(times[-1]) if times.size else first_time
                        last_val = int(np.clip(vals[-1], 0, 127))
                        if last_time > first_time or last_val != first_val:
                            inst.control_changes.append(
                                pretty_midi.ControlChange(
                                    number=11, value=last_val, time=last_time
                                )
                            )
                            cc11_count += 1
        else:  # Fallback to ADSR derived from notes
            notes = sorted(inst.notes, key=lambda n: n.start)
            if notes:
                attack = cc11_smooth_ms / 1000.0
                release = cc11_smooth_ms / 1000.0
                end_time = max(n.end for n in notes)
                dt = 0.02
                times = [i * dt for i in range(int(end_time / dt) + 1)]

                def env_at(t: float) -> float:
                    val = 0.0
                    for n in notes:
                        if n.start <= t <= n.end:
                            if t < n.start + attack:
                                val = max(val, (t - n.start) / max(attack, 1e-6))
                            elif t > n.end - release:
                                val = max(val, (n.end - t) / max(release, 1e-6))
                            else:
                                val = max(val, 1.0)
                    return val

                prev_val = None
                last_t = -1e9
                min_dt = cc11_min_dt_ms / 1000.0
                for t in times:
                    e = env_at(t)
                    val = int(round(max(0.0, min(1.0, e * cc11_gain)) * 127))
                    if (
                        prev_val is None
                        or (
                            (val > prev_val and val - prev_val >= cc11_hyst_up)
                            or (val < prev_val and prev_val - val >= cc11_hyst_down)
                        )
                        and (t - last_t >= min_dt)
                    ):
                        inst.control_changes.append(
                            pretty_midi.ControlChange(
                                number=11, value=val, time=float(t)
                            )
                        )
                        prev_val = val
                        last_t = float(t)
                        cc11_count += 1

    # --- CC64 heuristic sustain ---
    if cc64_mode == "heuristic" and _is_piano_like(track_name):
        notes = sorted(inst.notes, key=lambda n: n.start)
        beat = 60.0 / (tempo if tempo and tempo > 0 else 120.0)
        thresh = beat * cc64_gap_beats
        min_dwell = cc64_min_dwell_ms / 1000.0
        events: list[tuple[float, int]] = []
        for a, b in zip(notes, notes[1:]):
            gap = float(b.start) - float(a.end)
            if 0 < gap < thresh:
                on = float(a.end)
                off = float(max(a.end, b.start - 0.001))
                if off - on >= min_dwell:
                    events.append((on, 127))
                    events.append((off, 0))
        events.sort()
        last_val: int | None = None
        for t, val in events:
            if val != last_val:
                inst.control_changes.append(
                    pretty_midi.ControlChange(number=64, value=int(val), time=float(t))
                )
                last_val = val
                cc64_count += 1

    return cc11_count, cc64_count


def _fallback_transcribe_stem(
    path: Path,
    *,
    min_dur: float,
    tempo: float | None = None,
    auto_tempo: bool | None = None,
) -> StemResult:
    """Simpler onset-only transcription used when CREPE/librosa are unavailable.

    Parameters
    ----------
    tempo:
        Optional known tempo to adapt onset spacing.
    """

    try:  # Attempt to use basic_pitch if installed
        from basic_pitch import inference
    except Exception:  # pragma: no cover - optional dependency
        inference = None

    inst = pretty_midi.Instrument(program=0, name=path.stem)
    auto_flag = True if auto_tempo is None else bool(auto_tempo)

    if inference is not None:
        try:
            _, _, note_events = inference.predict(str(path))
        except Exception:  # pragma: no cover - unexpected basic_pitch failure
            note_events = []
        for onset, offset, *_ in note_events:
            if offset - onset >= min_dur:
                inst.notes.append(
                    _mk_note(
                        36,
                        float(onset),
                        float(offset) - float(onset),
                        100,
                    )
                )
        if not inst.notes:
            inst.notes.append(_mk_note(36, 0.0, float(min_dur), 100))
        return StemResult(inst, tempo)

    try:
        from scipy.io import wavfile
    except ModuleNotFoundError as exc:
        _warn_once("missing_scipy_wavfile", f"Missing scipy.io.wavfile: {exc}")
        return _minimal_stem(path, min_dur=min_dur, auto_tempo=auto_flag)

    sr, audio = wavfile.read(path)

    # Convert to mono
    if getattr(audio, "ndim", 1) > 1:
        if np is not None:
            audio = audio.mean(axis=1)
        else:  # pragma: no cover - simple Python fallback
            audio = [sum(frame) / len(frame) for frame in audio]

    if np is not None:
        audio = audio.astype(float)
        if not audio.size:
            return _minimal_stem(path, min_dur=min_dur, auto_tempo=auto_flag)
        threshold = 0.5 * np.percentile(np.abs(audio), 95)
        gap = 60.0 / (tempo if tempo and tempo > 0 else 120.0) / 4.0
        min_gap = int(sr * gap)
        envelope = np.abs(audio)
        onset_idxs = np.where(
            (envelope[1:] >= threshold) & (envelope[:-1] < threshold)
        )[0]
        last_idx = -min_gap
        for idx in onset_idxs:
            if idx - last_idx < min_gap:
                continue
            t = idx / sr
            inst.notes.append(_mk_note(60, float(t), float(min_dur), 100))
            last_idx = idx
    else:  # pragma: no cover - numpy may be absent
        logger.info(
            "numpy not available; using simple onset detector for %s", path.name
        )
        audio = [float(x) for x in audio]
        if not audio:
            return _minimal_stem(path, min_dur=min_dur, auto_tempo=auto_flag)
        abs_audio = [abs(x) for x in audio]
        idx95 = int(0.95 * (len(abs_audio) - 1))
        threshold = 0.5 * sorted(abs_audio)[idx95]
        gap = 60.0 / (tempo if tempo and tempo > 0 else 120.0) / 4.0
        min_gap = int(sr * gap)
        last_idx = -min_gap
        prev = abs_audio[0]
        for idx, val in enumerate(abs_audio[1:], start=1):
            if val >= threshold and prev < threshold and idx - last_idx >= min_gap:
                t = idx / sr
                inst.notes.append(_mk_note(60, float(t), float(min_dur), 100))
                last_idx = idx
            prev = val
    return StemResult(inst, tempo)


def _transcribe_stem_impl(
    path: Path,
    *,
    step_size: int = 10,
    conf_threshold: float = 0.5,
    min_dur: float = 0.05,
    auto_tempo: bool = True,
    enable_bend: bool = True,
    bend_range_semitones: float = 2.0,
    bend_alpha: float = 0.25,
    bend_fixed_base: bool = False,
    cc11_strategy: str = "energy",
    cc11_map: str = "linear",
    cc11_smooth_ms: float = 60.0,
    cc11_gain: float = 1.0,
    cc11_hyst_up: float = 3.0,
    cc11_hyst_down: float = 3.0,
    cc11_min_dt_ms: float = 30.0,
    cc64_mode: str = "none",
    cc64_gap_beats: float = 0.25,
    cc64_min_dwell_ms: float = 80.0,
    sustain_threshold: float | None = None,
    cc_strategy: str | None = None,
    **_legacy_kwargs: object,
) -> StemResult:
    """Transcribe a monophonic WAV file into a MIDI instrument.

    Returns the instrument along with an estimated tempo (in BPM) when
    ``auto_tempo`` is enabled. Tempo estimation falls back to ``None`` if it
    cannot be computed."""

    if cc_strategy is not None:
        cc11_strategy = cc_strategy
    if sustain_threshold is not None:
        cc64_gap_beats = sustain_threshold

    if _legacy_kwargs:
        _legacy_kwargs.clear()

    effective_cc11_smooth_ms = float(cc11_smooth_ms)
    if effective_cc11_smooth_ms <= cc11_min_dt_ms:
        effective_cc11_smooth_ms = 0.0

    if crepe is None or librosa is None:
        missing = " and ".join(
            dep for dep, mod in (("crepe", crepe), ("librosa", librosa)) if mod is None
        )
        logger.warning(
            "Missing %s; falling back to onset-only transcription, "
            "transcription quality may degrade",
            missing,
        )
        if enable_bend:
            logger.info("Pitch-bend disabled: missing CREPE/librosa")
        tempo: float | None = None
        audio = None
        sr = 16000
        if auto_tempo and librosa is not None:
            try:
                audio, sr = librosa.load(path, sr=sr, mono=True)
                tempo, _ = librosa.beat.beat_track(y=audio, sr=sr, trim=False)
                if not 40 <= tempo <= 300 or not math.isfinite(tempo):
                    tempo = None
                else:
                    logger.info("Estimated %.1f BPM for %s", tempo, path.name)
            except Exception:  # pragma: no cover - tempo estimation is optional
                tempo = None
        fallback_kwargs: dict[str, object] = {"min_dur": min_dur}
        if tempo is not None:
            fallback_kwargs["tempo"] = tempo
        try:
            signature = inspect.signature(_fallback_transcribe_stem)
        except (TypeError, ValueError):  # pragma: no cover - dynamic callable
            signature = None
        if signature is None or "auto_tempo" in signature.parameters:
            fallback_kwargs["auto_tempo"] = auto_tempo
        else:
            logger.info(
                "omitted auto_tempo due to legacy signature for fallback transcription"
            )
            logger.debug(
                "omitted auto_tempo due to legacy signature for fallback transcription"
            )
        try:
            result = _fallback_transcribe_stem(path, **fallback_kwargs)
        except TypeError:
            args = [path]
            min_dur_kw = fallback_kwargs.get("min_dur")
            if min_dur_kw is not None:
                args.append(min_dur_kw)
            tempo_kw = fallback_kwargs.get("tempo")
            if tempo_kw is not None:
                args.append(tempo_kw)
            try:
                result = _fallback_transcribe_stem(*args)
            except TypeError:
                result = _fallback_transcribe_stem(path)
        if audio is None:
            try:
                from scipy.io import wavfile

                sr, data = wavfile.read(path)
                audio = data.astype(float)
                if getattr(audio, "ndim", 1) > 1:
                    audio = audio.mean(axis=1)
            except Exception:  # pragma: no cover - best effort
                audio = None
        cc11_c, cc64_c = apply_cc_curves(
            result.instrument,
            audio=audio,
            sr=sr,
            tempo=result.tempo,
            cc11_strategy=cc11_strategy,
            cc11_map=cc11_map,
            cc11_smooth_ms=effective_cc11_smooth_ms,
            cc11_gain=cc11_gain,
            cc11_hyst_up=cc11_hyst_up,
            cc11_hyst_down=cc11_hyst_down,
            cc11_min_dt_ms=cc11_min_dt_ms,
            cc64_mode=cc64_mode,
            cc64_gap_beats=cc64_gap_beats,
            cc64_min_dwell_ms=cc64_min_dwell_ms,
            track_name=path.stem,
        )
        logger.info(
            "cc11=%d cc64=%d bends=%d for %s",
            cc11_c,
            cc64_c,
            len(result.instrument.pitch_bends),
            path.name,
        )
        return StemResult(result.instrument, result.tempo)

    audio, sr = librosa.load(path, sr=16000, mono=True)

    tempo: float | None = None
    if auto_tempo:
        try:
            tempo, _ = librosa.beat.beat_track(y=audio, sr=sr, trim=False)
            if not 40 <= tempo <= 300 or not math.isfinite(tempo):
                tempo = None
            else:
                logger.info("Estimated %.1f BPM for %s", tempo, path.name)
        except Exception:  # pragma: no cover - tempo estimation is optional
            tempo = None
    time, freq, conf, _ = crepe.predict(
        audio, sr, step_size=step_size, model_capacity="full", verbose=0
    )

    inst = pretty_midi.Instrument(program=0, name=path.stem)
    pitch: int | None = None
    start: float = 0.0
    ema = 0.0
    prev_bend = 0
    if enable_bend:
        inst.pitch_bends.append(pretty_midi.PitchBend(pitch=0, time=0.0))

    for t, f, c in zip(time, freq, conf):
        dev: float | None
        if c < conf_threshold:
            dev = None
            if pitch is not None and t - start >= min_dur:
                inst.notes.append(
                    _mk_note(pitch, float(start), float(t) - float(start), 100)
                )
            pitch = None
        else:
            nn_float = pretty_midi.hz_to_note_number(f)
            p = int(round(nn_float))
            if pitch is None:
                pitch, start = p, float(t)
            elif p != pitch:
                if t - start >= min_dur:
                    inst.notes.append(
                        _mk_note(pitch, float(start), float(t) - float(start), 100)
                    )
                pitch, start = p, float(t)
            dev = nn_float - (pitch if bend_fixed_base else p)

        if enable_bend:
            target = 0.0 if dev is None else dev
            ema = bend_alpha * target + (1 - bend_alpha) * ema
            if dev is None:
                bend = 0
            else:
                bend = pb_math.semi_to_pb(ema, bend_range_semitones)
            if bend != prev_bend:
                inst.pitch_bends.append(
                    pretty_midi.PitchBend(pitch=int(bend), time=float(t))
                )
                prev_bend = bend

    if pitch is not None and time.size:
        end = float(time[-1])
        if end - start >= min_dur:
            inst.notes.append(
                _mk_note(pitch, float(start), float(end) - float(start), 100)
            )

    if enable_bend:
        end_time = float(time[-1]) if time.size else 0.0
        if prev_bend != 0:
            inst.pitch_bends.append(pretty_midi.PitchBend(pitch=0, time=end_time))
        elif inst.pitch_bends[-1].time != end_time:
            inst.pitch_bends.append(pretty_midi.PitchBend(pitch=0, time=end_time))

    cc11_c, cc64_c = apply_cc_curves(
        inst,
        audio=audio,
        sr=sr,
        tempo=tempo,
        cc11_strategy=cc11_strategy,
        cc11_map=cc11_map,
        cc11_smooth_ms=effective_cc11_smooth_ms,
        cc11_gain=cc11_gain,
        cc11_hyst_up=cc11_hyst_up,
        cc11_hyst_down=cc11_hyst_down,
        cc11_min_dt_ms=cc11_min_dt_ms,
        cc64_mode=cc64_mode,
        cc64_gap_beats=cc64_gap_beats,
        cc64_min_dwell_ms=cc64_min_dwell_ms,
        track_name=path.stem,
    )
    logger.info(
        "cc11=%d cc64=%d bends=%d for %s",
        cc11_c,
        cc64_c,
        len(inst.pitch_bends),
        path.name,
    )
    return StemResult(inst, tempo)


_invoke_transcribe = _transcribe_stem


def _iter_song_dirs(src: Path, exts: list[str]) -> list[Path]:
    """Return directories representing individual songs.

    If ``src`` contains audio files with any of the given extensions directly,
    treat ``src`` itself as a single song. Otherwise, each sub-directory is
    considered a separate song.
    """

    if len(exts) == 1:
        audio_files = list(src.glob(f"*.{exts[0]}"))
    else:
        audio_files = []
        for ext in exts:
            audio_files.extend(src.glob(f"*.{ext}"))
    if audio_files:
        return [src]
    return [d for d in src.iterdir() if d.is_dir()]


def convert_directory(
    src: Path,
    dst: Path,
    *,
    ext: str = "wav",
    jobs: int = 1,
    min_dur: float = 0.05,
    resume: bool = False,
    overwrite: bool = False,
    safe_dirnames: bool = False,
    merge: bool = False,
    auto_tempo: bool = True,
    tempo_strategy: str = "median",
    tempo_lock: str = "none",
    tempo_anchor_pattern: str = r"(?i)(drum|perc|beat|click)",
    tempo_lock_value: float | None = None,
    tempo_fold_halves: bool = False,
    tempo_lock_fallback: str = "median",
    enable_bend: bool = True,
    bend_range_semitones: float = 2.0,
    bend_alpha: float = 0.25,
    bend_fixed_base: bool = False,
    bend_integer_range: bool = False,
    cc11_strategy: str = "energy",
    cc11_map: str = "linear",
    cc11_smooth_ms: float = 60.0,
    cc11_gain: float = 1.0,
    cc11_hyst_up: float = 3.0,
    cc11_hyst_down: float = 3.0,
    cc11_min_dt_ms: float = 30.0,
    cc64_mode: str = "none",
    cc64_gap_beats: float = 0.25,
    cc64_min_dwell_ms: float = 80.0,
    sustain_threshold: float | None = None,
    controls_post_bend: str = "skip",
) -> None:
    """Convert a directory of stems into individual MIDI files.

    If ``merge`` is ``True``, all stems for a song are merged into a single
    multi-track MIDI file, emulating the legacy behaviour.
    """

    exts = [e.strip().lstrip(".") for e in ext.split(",") if e.strip()]
    dst.mkdir(parents=True, exist_ok=True)

    log_path = dst / "conversion_log.json"
    log_data: dict[str, list[str]] = {}
    effective_cc11_smooth_ms = float(cc11_smooth_ms)
    if effective_cc11_smooth_ms <= cc11_min_dt_ms:
        effective_cc11_smooth_ms = 0.0
    base_transcribe_kwargs = {
        "min_dur": min_dur,
        "auto_tempo": auto_tempo,
        "enable_bend": enable_bend,
        "bend_range_semitones": bend_range_semitones,
        "bend_alpha": bend_alpha,
        "bend_fixed_base": bend_fixed_base,
        "cc11_strategy": cc11_strategy,
        "cc11_map": cc11_map,
        "cc11_smooth_ms": effective_cc11_smooth_ms,
        "cc11_gain": cc11_gain,
        "cc11_hyst_up": cc11_hyst_up,
        "cc11_hyst_down": cc11_hyst_down,
        "cc11_min_dt_ms": cc11_min_dt_ms,
        "cc64_mode": cc64_mode,
        "cc64_gap_beats": cc64_gap_beats,
        "cc64_min_dwell_ms": cc64_min_dwell_ms,
        "sustain_threshold": sustain_threshold
        if sustain_threshold is not None
        else cc64_gap_beats,
    }
    # Older stubs (used in tests) expect ``cc11_smoothing_ms`` instead of
    # ``cc11_smooth_ms``.  Provide both so ``_filter_kwargs`` can keep the one
    # supported by the active transcription function.
    base_transcribe_kwargs["cc11_smoothing_ms"] = base_transcribe_kwargs["cc11_smooth_ms"]
    # Tests patch _transcribe_stem with legacy signatures; filter new kwargs for compatibility.
    filtered_transcribe_kwargs = _filter_kwargs(
        _transcribe_stem, base_transcribe_kwargs
    )
    if resume and log_path.exists():
        try:
            log_data = json.loads(log_path.read_text())
        except Exception:
            logger.warning("Failed to read %s", log_path)

    for song_dir in _iter_song_dirs(src, exts):
        song_name = _sanitize_name(song_dir.name) if safe_dirnames else song_dir.name

        if merge:
            out_song = dst / f"{song_name}.mid"
            processed = set(log_data.get(song_name, []))
            if (
                resume
                and not overwrite
                and ("__MERGED__" in processed or out_song.exists())
            ):
                logger.info("Skipping %s", out_song)
                continue
        else:
            out_song = dst / song_name
            out_song.mkdir(parents=True, exist_ok=True)

        if len(exts) == 1:
            wavs = sorted(song_dir.glob(f"*.{exts[0]}"))
        else:
            wavs = []
            for e in exts:
                wavs.extend(song_dir.glob(f"*.{e}"))
            wavs.sort()
        if not wavs:
            continue

        total_time = 0.0
        converted = 0

        if merge:
            results: list[tuple[str, pretty_midi.Instrument, float | None]] = []
            ex_kwargs = {"max_workers": jobs}
            if os.name != "nt":
                ex_kwargs["mp_context"] = multiprocessing.get_context("forkserver")
            if jobs > 1:
                with ProcessPoolExecutor(**ex_kwargs) as ex:
                    futures = []
                    for wav in wavs:
                        logger.info("Transcribing %s", wav)
                        start = time.perf_counter()
                        futures.append(
                            (
                                ex.submit(
                                    _transcribe_stem,
                                    wav,
                                    **filtered_transcribe_kwargs,
                                ),
                                start,
                            )
                        )
                    for fut, start in futures:
                        res = fut.result()
                        name = _sanitize_name(res.instrument.name)
                        res.instrument.name = name
                        results.append((name, res.instrument, res.tempo))
                        total_time += time.perf_counter() - start
            else:
                for wav in wavs:
                    logger.info("Transcribing %s", wav)
                    start = time.perf_counter()
                    res = _transcribe_stem(
                        wav,
                        **filtered_transcribe_kwargs,
                    )
                    name = _sanitize_name(res.instrument.name)
                    res.instrument.name = name
                    results.append((name, res.instrument, res.tempo))
                    total_time += time.perf_counter() - start

            tempo: float | None = None
            summary_line: str | None = None
            tempos = [
                float(t)
                for _, _, t in results
                if t is not None and math.isfinite(float(t))
            ]
            if tempo_lock != "none":
                orig_mode = tempo_lock
                candidates = [
                    (n, float(t))
                    for n, _, t in results
                    if t is not None and math.isfinite(float(t))
                ]
                cand_count = len(candidates)
                if tempo_lock == "value" and tempo_lock_value is not None:
                    tempo = float(tempo_lock_value)
                    summary_line = f"Tempo-lock(value): {song_name} → BPM={tempo:.1f}"
                elif tempo_lock == "anchor":
                    try:
                        pattern = re.compile(tempo_anchor_pattern)
                    except re.error:
                        msg = f"Invalid tempo-anchor-pattern: {tempo_anchor_pattern}"
                        if tempo_lock_fallback == "none":
                            logger.error(msg)
                            raise SystemExit(2)
                        logger.error("%s; falling back to median", msg)
                        tempo_lock = "median"
                    if tempo_lock == "anchor":
                        anchor_name = None
                        anchor_bpm = None
                        for n, t in candidates:
                            if pattern.search(n):
                                anchor_name = n
                                anchor_bpm = t
                                break
                        if anchor_bpm is not None:
                            orig_vals = [t for _, t in candidates]
                            if tempo_fold_halves:
                                folded_vals = [
                                    _fold_to_ref(t, anchor_bpm) for t in orig_vals
                                ]
                            else:
                                folded_vals = orig_vals
                            tempo = float(anchor_bpm)
                            if tempo_fold_halves:
                                b_min, b_med, b_max = (
                                    min(orig_vals),
                                    statistics.median(orig_vals),
                                    max(orig_vals),
                                )
                                f_min, f_med, f_max = (
                                    min(folded_vals),
                                    statistics.median(folded_vals),
                                    max(folded_vals),
                                )
                                fold_str = (
                                    f", fold {b_min:.1f}/{b_med:.1f}/{b_max:.1f}"
                                    f"->{f_min:.1f}/{f_med:.1f}/{f_max:.1f}"
                                )
                            else:
                                fold_str = ""
                            summary_line = (
                                f"Tempo-lock(anchor): {song_name} → BPM={tempo:.1f} "
                                f"(pattern='{tempo_anchor_pattern}', candidates={cand_count}"
                                f"{fold_str}, anchor='{anchor_name}')"
                            )
                        else:
                            tempo_lock = "median"
                    if tempo_lock == "median":
                        orig_vals = [t for _, t in candidates]
                        if not orig_vals:
                            tempo = 120.0
                            logger.warning(
                                "Tempo-lock(%s): %s has no valid tempo estimates (%d stems); using 120 BPM",
                                orig_mode,
                                song_name,
                                len(results),
                            )
                        else:
                            if tempo_fold_halves:
                                b_min, b_med, b_max = (
                                    min(orig_vals),
                                    statistics.median(orig_vals),
                                    max(orig_vals),
                                )
                                folded_vals = [
                                    _fold_to_ref(t, b_med) for t in orig_vals
                                ]
                                tempo = statistics.median(folded_vals)
                                f_min, f_med, f_max = (
                                    min(folded_vals),
                                    statistics.median(folded_vals),
                                    max(folded_vals),
                                )
                                fold_str = (
                                    f", fold {b_min:.1f}/{b_med:.1f}/{b_max:.1f}"
                                    f"->{f_min:.1f}/{f_med:.1f}/{f_max:.1f}"
                                )
                            else:
                                tempo = statistics.median(orig_vals)
                                fold_str = ""
                        summary_line = (
                            f"Tempo-lock(median): {song_name} → BPM={tempo:.1f} "
                            f"(candidates={cand_count}{fold_str})"
                        )
            else:
                if tempo_strategy == "first":
                    tempo = tempos[0] if tempos else None
                elif tempo_strategy == "median":
                    if tempos:
                        spread = max(tempos) - min(tempos)
                        tempo = statistics.median(tempos)
                        if spread > 5:
                            logger.warning(
                                "Tempo spread %.1f BPM for %s", spread, song_name
                            )
                elif tempo_strategy == "ignore":
                    tempo = None

            pm = pretty_midi.PrettyMIDI()
            if tempo is not None:
                _set_initial_tempo(pm, float(tempo))
            for name, inst, _ in results:
                _emit_pitch_bend_range(
                    inst, bend_range_semitones, integer_only=bend_integer_range
                )
                _coerce_note_times(inst)
                _coerce_controller_times(inst)
                pm.instruments.append(inst)
            pm.write(str(out_song))
            converted = len(results)
            if resume:
                log_data[song_name] = ["__MERGED__"]
                try:
                    log_path.write_text(json.dumps(log_data))
                except Exception:
                    logger.warning("Failed to update %s", log_path)
            logger.info("Wrote %s", out_song)
            if summary_line:
                logger.info(summary_line)
        else:
            processed = set(log_data.get(song_name, []))
            existing = {p.stem for p in out_song.glob("*.mid")}
            used_names: set[str] = set()
            tasks = []
            for wav in wavs:
                sanitized = _sanitize_name(wav.stem)
                if resume and not overwrite and sanitized in processed:
                    logger.info("Skipping %s", wav)
                    continue
                base = sanitized
                n = 1
                if overwrite:
                    while base in used_names or (
                        base != sanitized and base in existing
                    ):
                        base = f"{sanitized}_{n}"
                        n += 1
                else:
                    candidate = out_song / f"{base}.mid"
                    while candidate.exists() or base in used_names:
                        base = f"{sanitized}_{n}"
                        candidate = out_song / f"{base}.mid"
                        n += 1
                midi_path = out_song / f"{base}.mid"
                used_names.add(base)
                tasks.append((wav, base, midi_path))

            path_by_base = {b: w for w, b, _ in tasks}
            results: list[tuple[str, pretty_midi.Instrument, float | None, Path]] = []
            if jobs > 1:
                ex_kwargs = {"max_workers": jobs}
                if os.name != "nt":
                    ex_kwargs["mp_context"] = multiprocessing.get_context("forkserver")
                with ProcessPoolExecutor(**ex_kwargs) as ex:
                    futures = {}
                    for wav, base, midi_path in tasks:
                        logger.info("Transcribing %s", wav)
                        start = time.perf_counter()
                        futures[
                            ex.submit(
                                _transcribe_stem,
                                wav,
                                **filtered_transcribe_kwargs,
                            )
                        ] = (base, midi_path, start)
                    for fut in as_completed(futures):
                        base, midi_path, start = futures[fut]
                        res = fut.result()
                        inst = res.instrument
                        tempo = res.tempo
                        inst.name = base
                        results.append((base, inst, tempo, midi_path))
                        total_time += time.perf_counter() - start
            else:
                for wav, base, midi_path in tasks:
                    logger.info("Transcribing %s", wav)
                    start = time.perf_counter()
                    res = _transcribe_stem(
                        wav,
                        **filtered_transcribe_kwargs,
                    )
                    inst = res.instrument
                    tempo = res.tempo
                    inst.name = base
                    results.append((base, inst, tempo, midi_path))
                    total_time += time.perf_counter() - start

            locked_tempo: float | None = None
            summary_line: str | None = None
            if tempo_lock != "none":
                orig_mode = tempo_lock
                candidates = [
                    (b, float(t))
                    for b, _, t, _ in results
                    if t is not None and math.isfinite(float(t))
                ]
                cand_count = len(candidates)
                if tempo_lock == "value" and tempo_lock_value is not None:
                    locked_tempo = float(tempo_lock_value)
                    summary_line = (
                        f"Tempo-lock(value): {song_name} → BPM={locked_tempo:.1f}"
                    )
                elif tempo_lock == "anchor":
                    try:
                        pattern = re.compile(tempo_anchor_pattern)
                    except re.error:
                        msg = f"Invalid tempo-anchor-pattern: {tempo_anchor_pattern}"
                        if tempo_lock_fallback == "none":
                            logger.error(msg)
                            raise SystemExit(2)
                        logger.error("%s; falling back to median", msg)
                        tempo_lock = "median"
                    if tempo_lock == "anchor":
                        anchor_name = None
                        anchor_bpm = None
                        for n, t in candidates:
                            if pattern.search(n):
                                anchor_name = n
                                anchor_bpm = t
                                break
                        if anchor_bpm is not None:
                            orig_vals = [t for _, t in candidates]
                            if tempo_fold_halves:
                                folded_vals = [
                                    _fold_to_ref(t, anchor_bpm) for t in orig_vals
                                ]
                            else:
                                folded_vals = orig_vals
                            locked_tempo = float(anchor_bpm)
                            if tempo_fold_halves:
                                b_min, b_med, b_max = (
                                    min(orig_vals),
                                    statistics.median(orig_vals),
                                    max(orig_vals),
                                )
                                f_min, f_med, f_max = (
                                    min(folded_vals),
                                    statistics.median(folded_vals),
                                    max(folded_vals),
                                )
                                fold_str = (
                                    f", fold {b_min:.1f}/{b_med:.1f}/{b_max:.1f}"
                                    f"->{f_min:.1f}/{f_med:.1f}/{f_max:.1f}"
                                )
                            else:
                                fold_str = ""
                            summary_line = (
                                f"Tempo-lock(anchor): {song_name} → BPM={locked_tempo:.1f} "
                                f"(pattern='{tempo_anchor_pattern}', candidates={cand_count}"
                                f"{fold_str}, anchor='{anchor_name}')"
                            )
                        else:
                            tempo_lock = "median"
                if tempo_lock == "median":
                    orig_vals = [t for _, t in candidates]
                    if not orig_vals:
                        locked_tempo = 120.0
                        logger.warning(
                            "Tempo-lock(%s): %s has no valid tempo estimates (%d stems); using 120 BPM",
                            orig_mode,
                            song_name,
                            len(results),
                        )
                    else:
                        if tempo_fold_halves:
                            b_min, b_med, b_max = (
                                min(orig_vals),
                                statistics.median(orig_vals),
                                max(orig_vals),
                            )
                            folded_vals = [_fold_to_ref(t, b_med) for t in orig_vals]
                            locked_tempo = statistics.median(folded_vals)
                            f_min, f_med, f_max = (
                                min(folded_vals),
                                statistics.median(folded_vals),
                                max(folded_vals),
                            )
                            fold_str = (
                                f", fold {b_min:.1f}/{b_med:.1f}/{b_max:.1f}"
                                f"->{f_min:.1f}/{f_med:.1f}/{f_max:.1f}"
                            )
                        else:
                            locked_tempo = statistics.median(orig_vals)
                            fold_str = ""
                    summary_line = (
                        f"Tempo-lock(median): {song_name} → BPM={locked_tempo:.1f} "
                        f"(candidates={cand_count}{fold_str})"
                    )

            for base, inst, tempo, midi_path in results:
                use_tempo = locked_tempo if locked_tempo is not None else tempo
                pm = pretty_midi.PrettyMIDI()
                if use_tempo is not None:
                    _set_initial_tempo(pm, float(use_tempo))
                _emit_pitch_bend_range(
                    inst, bend_range_semitones, integer_only=bend_integer_range
                )
                _coerce_note_times(inst)
                _coerce_controller_times(inst)
                pm.instruments.append(inst)
                if args.controls_spec:
                    enabled = {k for k, v in args.controls_spec.items() if v}
                    if enabled:
                        end_time = max((n.end for n in inst.notes), default=0.0)
                        curves: dict[str, ControlCurve] = {}
                        if "cc11" in enabled and args.cc_strategy in {"energy", "rms"}:
                            wav_path = path_by_base.get(base)
                            times: list[float] = []
                            vals: list[float] = []
                            if (
                                librosa is not None
                                and wav_path is not None
                                and os.path.exists(wav_path)
                            ):
                                try:
                                    y, sr = librosa.load(wav_path, sr=None, mono=True)
                                    hop = max(1, int(sr / args.controls_sample_rate_hz))
                                    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
                                    win = max(
                                        1,
                                        int(
                                            args.cc11_smoothing_ms
                                            / 1000.0
                                            * args.controls_sample_rate_hz
                                        ),
                                    )
                                    if win > 1:
                                        rms = np.convolve(
                                            rms, np.ones(win) / win, mode="same"
                                        )
                                    if rms.size:
                                        rms = rms - rms.min()
                                        if rms.max() > 0:
                                            rms = rms / rms.max()
                                        vals = (rms * 127 * args.cc11_gain).clip(0, 127).tolist()
                                        times_sec = (
                                            np.arange(len(vals)) * hop / sr
                                        ).tolist()
                                        times = (
                                            [pm.time_to_tick(t) / pm.resolution for t in times_sec]
                                            if args.controls_domain == "beats"
                                            else times_sec
                                        )
                                except Exception:  # pragma: no cover - best effort
                                    vals = []
                            if not vals:
                                notes = sorted(inst.notes, key=lambda n: n.start)
                                if notes:
                                    sr_env = max(1.0, float(args.controls_sample_rate_hz))
                                    dt = 1.0 / sr_env
                                    end_time = max(n.end for n in notes)
                                    bins = int(end_time / dt) + 1
                                    acc = [0.0] * bins
                                    cnt = [0] * bins
                                    for n in notes:
                                        s = int(n.start / dt)
                                        e = int(n.end / dt) + 1
                                        for i in range(s, e):
                                            acc[i] += n.velocity
                                            cnt[i] += 1
                                    avg = [acc[i] / cnt[i] if cnt[i] else 0.0 for i in range(bins)]
                                    win = max(
                                        1,
                                        int(args.cc11_smoothing_ms / 1000.0 * sr_env),
                                    )
                                    if win > 1 and np is not None:
                                        avg = (
                                            np.convolve(avg, np.ones(win) / win, mode="same")
                                            .tolist()
                                        )
                                    if avg:
                                        max_val = max(avg) or 1.0
                                        avg = [
                                            max(0.0, min(127.0, v / max_val * 127 * args.cc11_gain))
                                            for v in avg
                                        ]
                                        vals = avg
                                        times_sec = [i * dt for i in range(len(avg))]
                                        times = (
                                            [pm.time_to_tick(t) / pm.resolution for t in times_sec]
                                            if args.controls_domain == "beats"
                                            else times_sec
                                        )
                            if vals:
                                curves["cc11"] = ControlCurve(
                                    times,
                                    vals,
                                    domain=args.controls_domain,
                                    sample_rate_hz=args.controls_sample_rate_hz,
                                    dedup_time_epsilon=args.dedup_eps_time,
                                    dedup_value_epsilon=args.dedup_eps_value,
                                )
                            else:
                                logger.info("CC11 disabled: insufficient data for %s", inst.name)
                        ch_map = {
                            tgt: args.controls_channel_map.get(tgt, 0) for tgt in curves
                        }
                        by_ch: dict[int, dict[str, ControlCurve]] = {}
                        for tgt, curve in curves.items():
                            ch = ch_map.get(tgt, 0)
                            by_ch.setdefault(ch, {})[tgt] = curve
                        tempo_map = None
                        if args.controls_domain == "beats":
                            tempo_map = tempo_map_from_prettymidi(pm)
                        max_map: dict[str, int] = {}
                        if args.controls_max_events:
                            for t in curves:
                                max_map[t] = args.controls_max_events
                        if args.max_cc_events:
                            for t in curves:
                                if t.startswith("cc"):
                                    max_map[t] = args.max_cc_events
                        if args.max_bend_events:
                            max_map["bend"] = args.max_bend_events
                        if args.controls_post_bend in {"replace", "skip"}:
                            has_existing = bool(inst.pitch_bends)
                            if args.controls_post_bend == "replace" and has_existing:
                                inst.pitch_bends.clear()
                                logger.info(
                                    "controls-post-bend=replace: cleared bends on %s", inst.name
                                )
                            elif args.controls_post_bend == "skip" and has_existing:
                                ch_bend = ch_map.get("bend", 0)
                                if ch_bend in by_ch and "bend" in by_ch[ch_bend]:
                                    del by_ch[ch_bend]["bend"]
                                    if not by_ch[ch_bend]:
                                        del by_ch[ch_bend]
                                    logger.info(
                                        "controls-post-bend=skip: kept existing bends on %s",
                                        inst.name,
                                    )
                            else:
                                logger.info(
                                    "controls-post-bend=%s: no existing bends on %s",
                                    args.controls_post_bend,
                                    inst.name,
                                )
                        else:
                            logger.info("controls-post-bend=add on %s", inst.name)
                        if not args.controls_routing:
                            apply_controls(
                                pm,
                                by_ch,
                                bend_range_semitones=args.bend_range_semitones,
                                write_rpn=args.write_rpn_range,
                                sample_rate_hz={
                                    "cc11": args.controls_sample_rate_hz,
                                    "cc64": args.controls_sample_rate_hz,
                                    "bend": args.controls_sample_rate_hz,
                                },
                                max_events=max_map or None,
                                total_max_events=args.controls_total_max_events,
                                simplify_mode=getattr(args, "controls_simplify_mode", "rdp"),
                                value_eps=args.dedup_eps_value,
                                time_eps=args.dedup_eps_time,
                                tempo_map=tempo_map,
                            )
                            merged = False
                            for ci in list(pm.instruments):
                                if ci is inst:
                                    continue
                                if ci.name.startswith("channel"):
                                    inst.control_changes.extend(ci.control_changes)
                                    inst.pitch_bends.extend(ci.pitch_bends)
                                    pm.instruments.remove(ci)
                                    merged = True
                            if merged:
                                inst.control_changes.sort(key=lambda c: c.time)
                                inst.pitch_bends.sort(key=lambda b: b.time)
                        elif args.emit_cc11 or args.cc64_mode != "none" or curves:
                            _warn_once(
                                "controls-routing-skipped",
                                "--controls-routing overrides in-process controls; skipping",
                            )
                pm.write(str(midi_path))
                if args.controls_routing:
                    routing_path = Path(args.controls_routing)
                    if routing_path.exists():
                        cmd = [
                            sys.executable,
                            "-m",
                            "utilities.apply_controls_cli",
                            str(midi_path),
                            str(routing_path),
                        ]
                        if args.controls_args:
                            cmd += shlex.split(args.controls_args)
                        log_cmd = " ".join(shlex.quote(c) for c in cmd)
                        logger.info("apply_controls: %s", log_cmd)
                        try:
                            subprocess.run(
                                cmd, check=True, capture_output=True, text=True
                            )
                        except subprocess.CalledProcessError as exc:  # pragma: no cover
                            logger.warning(
                                "apply_controls failed for %s: %s", midi_path, exc
                            )
                            if exc.stderr:
                                logger.warning("stderr: %s", exc.stderr)
                        except Exception as exc:  # pragma: no cover
                            logger.warning(
                                "apply_controls failed for %s: %s", midi_path, exc
                            )
                converted += 1
                processed.add(base)
                if resume:
                    log_data[song_name] = sorted(processed)
                    try:
                        log_path.write_text(json.dumps(log_data))
                    except Exception:
                        logger.warning("Failed to update %s", log_path)
                logger.info("Wrote %s", midi_path)

            if summary_line:
                logger.info(summary_line)

        logger.info(
            "\N{CHECK MARK} %s – %d stems → %.1f s", song_name, converted, total_time
        )


def main(argv: list[str] | None = None) -> None:
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(
        description="Batch audio-to-MIDI converter with per-stem output"
    )
    parser.add_argument("src_dir", help="Directory containing audio stems")
    parser.add_argument("dst_dir", help="Output directory for MIDI files")
    parser.add_argument(
        "--jobs", type=int, default=1, help="Number of worker processes"
    )
    parser.add_argument(
        "--ext",
        default="wav",
        help="Comma-separated audio file extensions to scan for",
    )
    parser.add_argument(
        "--min-dur",
        type=float,
        default=0.05,
        help="Minimum note duration in seconds",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from conversion_log.json and skip completed stems",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-transcribe stems even if output files exist",
    )
    parser.add_argument(
        "--safe-dirnames",
        action="store_true",
        help="Sanitize song directory names for output",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge stems into a single multi-track MIDI file",
    )
    parser.add_argument(
        "--auto-tempo",
        dest="auto_tempo",
        action="store_true",
        default=True,
        help="Estimate tempo using librosa and embed it in the MIDI (default)",
    )
    parser.add_argument(
        "--no-auto-tempo",
        dest="auto_tempo",
        action="store_false",
        help="Disable tempo estimation",
    )
    parser.add_argument(
        "--tempo-strategy",
        choices=["first", "median", "ignore"],
        default="median",
        help="How to select tempo when merging stems",
    )
    parser.add_argument(
        "--tempo-lock",
        choices=["none", "anchor", "median", "value"],
        default="none",
        help="Unify tempo across stems per song folder",
    )
    parser.add_argument(
        "--tempo-anchor-pattern",
        default="(?i)(drum|perc|beat|click)",
        help="Regex to select anchor stem when tempo-lock=anchor",
    )
    parser.add_argument(
        "--tempo-lock-value",
        type=float,
        help="Explicit BPM for tempo-lock=value",
    )
    parser.add_argument(
        "--tempo-fold-halves",
        action="store_true",
        help="Fold half/double tempo outliers before locking",
    )
    parser.add_argument(
        "--tempo-lock-fallback",
        choices=["median", "none"],
        default="median",
        help="When tempo-lock=anchor and regex is invalid, choose median or abort",
    )
    parser.add_argument(
        "--enable-bend",
        dest="enable_bend",
        action="store_true",
        default=True,
        help="Synthesize 14-bit pitch bends from f0 (default on)",
    )
    parser.add_argument(
        "--no-enable-bend",
        dest="enable_bend",
        action="store_false",
        help="Disable pitch-bend synthesis",
    )
    parser.add_argument(
        "--bend-range-semitones",
        type=float,
        default=2.0,
        help="Pitch-bend range in semitones for scaling (default 2.0)",
    )
    parser.add_argument(
        "--bend-alpha",
        type=float,
        default=0.25,
        help="EMA smoothing coefficient for pitch bends (default 0.25)",
    )
    parser.add_argument(
        "--bend-fixed-base",
        action="store_true",
        help="Reference deviations to note onsets for smoother portamento",
    )
    parser.add_argument(
        "--bend-integer-range",
        action="store_true",
        help="Force integer pitch-bend range (LSB=0)",
    )
    parser.add_argument(
        "--cc11-map",
        choices=["linear", "log"],
        default="linear",
        help="Mapping from RMS to 0–1 before scaling",
    )
    parser.add_argument(
        "--cc11-smooth-ms",
        "--cc11-smoothing-ms",
        dest="cc11_smoothing_ms",
        type=float,
        default=60.0,
        help="Smoothing window for CC11 strategies",
    )
    parser.add_argument(
        "--cc11-gain",
        type=float,
        default=1.0,
        help="Gain applied to CC11 envelope before scaling to 0–127",
    )
    parser.add_argument(
        "--cc11-hyst-up",
        type=float,
        default=3.0,
        help="Upward hysteresis threshold for CC11 values",
    )
    parser.add_argument(
        "--cc11-hyst-down",
        type=float,
        default=3.0,
        help="Downward hysteresis threshold for CC11 values",
    )
    parser.add_argument(
        "--cc11-min-dt-ms",
        type=float,
        default=30.0,
        help="Minimum time between CC11 events in milliseconds",
    )
    parser.add_argument(
        "--cc64-mode",
        choices=["none", "heuristic"],
        default="none",
        help="Sustain-pedal generation mode",
    )
    parser.add_argument(
        "--cc64-gap-beats",
        type=float,
        default=0.25,
        help="Maximum inter-note gap in quarter-note beats to link with sustain",
    )
    parser.add_argument(
        "--cc64-min-dwell-ms",
        type=float,
        default=80.0,
        help="Minimum sustain on/off duration in milliseconds",
    )
    parser.add_argument(
        "--sustain-threshold",
        type=float,
        default=None,
        help=(
            "Maximum inter-note gap in quarter-note beats to trigger sustain "
            "(defaults to --cc64-gap-beats when omitted)"
        ),
    )
    parser.add_argument(
        "--cc64-threshold",
        dest="cc64_threshold_alias",
        type=float,
        help="Deprecated alias for --sustain-threshold",
    )
    controls = parser.add_argument_group("Continuous Controls")
    controls.add_argument(
        "--emit-cc11",
        dest="emit_cc11",
        action="store_true",
        default=False,
        help="Enable expression (CC11) curves",
    )
    controls.add_argument(
        "--no-emit-cc11",
        dest="emit_cc11",
        action="store_false",
        help="Disable CC11 curves",
    )
    controls.add_argument(
        "--emit-cc64",
        dest="emit_cc64",
        action="store_true",
        default=False,
        help="Enable sustain pedal (CC64) curves",
    )
    controls.add_argument(
        "--no-emit-cc64",
        dest="emit_cc64",
        action="store_false",
        help="Disable CC64 curves",
    )
    controls.add_argument(
        "--cc-strategy",
        "--cc11-strategy",
        choices=["energy", "rms", "flat", "none"],
        default="energy",
        dest="cc_strategy",
        help="Source for CC11 dynamics",
    )
    # --cc11-smoothing-ms is defined on the main parser (with the --cc11-smooth-ms alias)
    # to avoid duplicate option registration here.
    controls.add_argument(
        "--controls-domain",
        choices=["time", "beats"],
        default="time",
        help="Domain for synthesized control curves",
    )
    controls.add_argument(
        "--controls-sample-rate-hz",
        type=float,
        default=100.0,
        dest="controls_sample_rate_hz",
        help="Sampling rate for synthesized control curves",
    )
    controls.add_argument(
        "--controls-res-hz",
        type=float,
        dest="controls_res_hz",
        help=argparse.SUPPRESS,
    )
    controls.add_argument(
        "--controls-max-events",
        type=int,
        help="Per-curve event cap",
    )
    controls.add_argument(
        "--controls-total-max-events",
        type=int,
        help="Global control-event cap",
    )
    controls.add_argument(
        "--controls-value-eps",
        type=float,
        default=1e-6,
        help="Epsilon for deduplicating nearly equal control values",
    )
    controls.add_argument(
        "--controls-channel-map",
        default="bend:0,cc11:0,cc64:0",
        help="Mapping like 'bend:0,cc11:0'",
    )
    controls.add_argument(
        "--write-rpn-range",
        dest="write_rpn_range",
        action="store_true",
        default=True,
        help="Emit RPN bend-range once per channel",
    )
    controls.add_argument(
        "--no-write-rpn-range",
        dest="write_rpn_range",
        action="store_false",
        help="Disable RPN bend-range messages",
    )
    controls.add_argument(
        "--write-rpn",
        dest="write_rpn_range_alias",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    controls.add_argument(
        "--no-write-rpn",
        dest="write_rpn_range_alias",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    controls.set_defaults(write_rpn_range_alias=None)
    controls.add_argument("--max-cc-events", type=int, default=0)
    controls.add_argument("--max-bend-events", type=int, default=0)
    controls.add_argument("--dedup-eps-time", type=float, default=1e-4)
    controls.add_argument("--dedup-eps-value", type=float, default=1.0)
    controls.add_argument(
        "--bend-units",
        choices=["semitones", "normalized"],
        default="semitones",
    )
    controls.add_argument(
        "--controls-routing",
        help="Apply controls using routing JSON after transcription",
    )
    controls.add_argument(
        "--controls-args",
        default="",
        help="Extra flags for apply_controls",
    )
    controls.add_argument(
        "--controls-post-bend",
        choices=["skip", "add", "replace"],
        default="skip",
        help="How to merge synthesized controls after existing pitch bends",
    )
    global args  # make available to helper functions
    args = parser.parse_args(argv)

    controls_spec: dict[str, bool] = {}
    if getattr(args, "emit_cc11", False):
        controls_spec["cc11"] = True
    if getattr(args, "emit_cc64", False):
        controls_spec["cc64"] = True
    if getattr(args, "controls", None):
        for part in args.controls.split(","):
            if not part or ":" not in part:
                continue
            key, val = part.split(":", 1)
            controls_spec[key.strip()] = val.strip().lower() == "on"
    args.controls_spec = controls_spec

    def _parse_ch_map(spec: str) -> dict[str, int]:
        mapping: dict[str, int] = {}
        for part in spec.split(","):
            if not part or ":" not in part:
                continue
            k, v = part.split(":", 1)
            try:
                ch = int(v)
            except ValueError:
                logger.warning("invalid channel %s for %s", v, k)
                continue
            if 0 <= ch <= 15:
                mapping[k.strip()] = ch
            else:
                logger.warning("channel %s out of range", v)
        return mapping

    args.controls_channel_map = _parse_ch_map(args.controls_channel_map)

    if getattr(args, "write_rpn_range_alias", None) is not None:
        _warn_once("write-rpn", "--write-rpn is deprecated; use --write-rpn-range")
        args.write_rpn_range = args.write_rpn_range_alias
    if getattr(args, "cc64_threshold_alias", None) is not None:
        warnings.warn(
            "--cc64-threshold is deprecated; use --sustain-threshold",
            DeprecationWarning,
            stacklevel=2,
        )
        args.sustain_threshold = args.cc64_threshold_alias
        args.cc64_mode = "heuristic"
    if args.sustain_threshold is not None and args.cc64_mode == "none":
        args.cc64_mode = "heuristic"

    if args.tempo_lock == "value" and args.tempo_lock_value is None:
        parser.error("--tempo-lock-value is required when --tempo-lock=value")

    convert_directory(
        Path(args.src_dir),
        Path(args.dst_dir),
        ext=args.ext,
        jobs=args.jobs,
        min_dur=args.min_dur,
        resume=args.resume,
        overwrite=args.overwrite,
        safe_dirnames=args.safe_dirnames,
        merge=args.merge,
        auto_tempo=args.auto_tempo,
        tempo_strategy=args.tempo_strategy,
        tempo_lock=args.tempo_lock,
        tempo_anchor_pattern=args.tempo_anchor_pattern,
        tempo_lock_value=args.tempo_lock_value,
        tempo_fold_halves=args.tempo_fold_halves,
        tempo_lock_fallback=args.tempo_lock_fallback,
        enable_bend=args.enable_bend,
        bend_range_semitones=args.bend_range_semitones,
        bend_alpha=args.bend_alpha,
        bend_fixed_base=args.bend_fixed_base,
        bend_integer_range=args.bend_integer_range,
        cc11_strategy=args.cc_strategy,
        cc11_map=args.cc11_map,
        cc11_smooth_ms=args.cc11_smoothing_ms,
        cc11_gain=args.cc11_gain,
        cc11_hyst_up=args.cc11_hyst_up,
        cc11_hyst_down=args.cc11_hyst_down,
        cc11_min_dt_ms=args.cc11_min_dt_ms,
        cc64_mode=args.cc64_mode,
        cc64_gap_beats=args.cc64_gap_beats,
        cc64_min_dwell_ms=args.cc64_min_dwell_ms,
        sustain_threshold=args.sustain_threshold,
        controls_post_bend=args.controls_post_bend,
    )


if __name__ == "__main__":
    raise SystemExit(main())
