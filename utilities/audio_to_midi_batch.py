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
import json
import logging
import math
import multiprocessing
import os
import re
import statistics
import time
import unicodedata
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

try:  # Optional dependency
    import numpy as np
except Exception:  # pragma: no cover - numpy may be absent
    np = None  # type: ignore
import pretty_midi

logger = logging.getLogger(__name__)

try:  # Optional heavy deps
    import crepe  # type: ignore
except Exception:  # pragma: no cover - handled by fallback
    crepe = None  # type: ignore

try:  # Optional heavy deps
    import librosa  # type: ignore
except Exception:  # pragma: no cover - handled by fallback
    librosa = None  # type: ignore


@dataclass(frozen=True)
class StemResult:
    instrument: pretty_midi.Instrument
    tempo: float | None


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
    inst: pretty_midi.Instrument, bend_range_semitones: float, *, t: float = 0.0
) -> None:
    """Insert RPN 0,0 events to set pitch-bend range."""
    msb = int(max(0, min(127, int(bend_range_semitones))))
    lsb = 64 if (bend_range_semitones - msb) >= 0.5 else 0
    cc = inst.control_changes
    cc.append(pretty_midi.ControlChange(number=101, value=0, time=float(t)))
    cc.append(pretty_midi.ControlChange(number=100, value=0, time=float(t)))
    cc.append(pretty_midi.ControlChange(number=6, value=msb, time=float(t)))
    cc.append(pretty_midi.ControlChange(number=38, value=lsb, time=float(t)))


def _fallback_transcribe_stem(
    path: Path, *, min_dur: float, tempo: float | None = None
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

    if inference is not None:
        try:
            _, _, note_events = inference.predict(str(path))
        except Exception:  # pragma: no cover - unexpected basic_pitch failure
            note_events = []
        for onset, offset, *_ in note_events:
            if offset - onset >= min_dur:
                inst.notes.append(
                    pretty_midi.Note(
                        velocity=100,
                        pitch=36,  # Kick drum placeholder until drum mapping is improved
                        start=float(onset),
                        end=float(offset),
                    )
                )
        return StemResult(inst, tempo)

    from scipy.io import wavfile

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
            return StemResult(inst, tempo)
        threshold = 0.5 * np.percentile(np.abs(audio), 95)
        gap = 60.0 / (tempo if tempo and tempo > 0 else 120.0) / 4.0
        min_gap = int(sr * gap)
        envelope = np.abs(audio)
        onset_idxs = np.where((envelope[1:] >= threshold) & (envelope[:-1] < threshold))[0]
        last_idx = -min_gap
        for idx in onset_idxs:
            if idx - last_idx < min_gap:
                continue
            t = idx / sr
            inst.notes.append(
                pretty_midi.Note(
                    velocity=100,
                    pitch=36,  # Kick drum placeholder until drum mapping is improved
                    start=float(t),
                    end=float(t + min_dur),
                )
            )
            last_idx = idx
    else:  # pragma: no cover - numpy may be absent
        logger.info("numpy not available; using simple onset detector for %s", path.name)
        audio = [float(x) for x in audio]
        if not audio:
            return StemResult(inst, tempo)
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
                inst.notes.append(
                    pretty_midi.Note(
                        velocity=100,
                        pitch=36,
                        start=float(t),
                        end=float(t + min_dur),
                    )
                )
                last_idx = idx
            prev = val
    return StemResult(inst, tempo)


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
) -> StemResult:
    """Transcribe a monophonic WAV file into a MIDI instrument.

    Returns the instrument along with an estimated tempo (in BPM) when
    ``auto_tempo`` is enabled. Tempo estimation falls back to ``None`` if it
    cannot be computed."""

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
        if auto_tempo and librosa is not None:
            try:
                audio, sr = librosa.load(path, sr=16000, mono=True)
                tempo, _ = librosa.beat.beat_track(y=audio, sr=sr, trim=False)
                if not 40 <= tempo <= 300 or not math.isfinite(tempo):
                    tempo = None
                else:
                    logger.info("Estimated %.1f BPM for %s", tempo, path.name)
            except Exception:  # pragma: no cover - tempo estimation is optional
                tempo = None
        return _fallback_transcribe_stem(path, min_dur=min_dur, tempo=tempo)

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
                    pretty_midi.Note(
                        velocity=100, pitch=pitch, start=start, end=float(t)
                    )
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
                        pretty_midi.Note(
                            velocity=100, pitch=pitch, start=start, end=float(t)
                        )
                    )
                pitch, start = p, float(t)
            dev = nn_float - (pitch if bend_fixed_base else p)

        if enable_bend:
            target = 0.0 if dev is None else dev
            ema = bend_alpha * target + (1 - bend_alpha) * ema
            if dev is None:
                bend = 0
            else:
                bend = int(
                    round(
                        max(-1.0, min(1.0, ema / bend_range_semitones)) * 8191
                    )
                )
            if bend != prev_bend:
                inst.pitch_bends.append(
                    pretty_midi.PitchBend(pitch=int(bend), time=float(t))
                )
                prev_bend = bend

    if pitch is not None and time.size:
        end = float(time[-1])
        if end - start >= min_dur:
            inst.notes.append(
                pretty_midi.Note(velocity=100, pitch=pitch, start=start, end=end)
            )

    if enable_bend:
        end_time = float(time[-1]) if time.size else 0.0
        if prev_bend != 0:
            inst.pitch_bends.append(
                pretty_midi.PitchBend(pitch=0, time=end_time)
            )
        elif inst.pitch_bends[-1].time != end_time:
            inst.pitch_bends.append(
                pretty_midi.PitchBend(pitch=0, time=end_time)
            )

    return StemResult(inst, tempo)


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
) -> None:
    """Convert a directory of stems into individual MIDI files.

    If ``merge`` is ``True``, all stems for a song are merged into a single
    multi-track MIDI file, emulating the legacy behaviour.
    """

    exts = [e.strip().lstrip(".") for e in ext.split(",") if e.strip()]
    dst.mkdir(parents=True, exist_ok=True)

    log_path = dst / "conversion_log.json"
    log_data: dict[str, list[str]] = {}
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
                            (ex.submit(
                                _transcribe_stem,
                                wav,
                                min_dur=min_dur,
                                auto_tempo=auto_tempo,
                                enable_bend=enable_bend,
                                bend_range_semitones=bend_range_semitones,
                                bend_alpha=bend_alpha,
                                bend_fixed_base=bend_fixed_base,
                            ), start)
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
                        min_dur=min_dur,
                        auto_tempo=auto_tempo,
                        enable_bend=enable_bend,
                        bend_range_semitones=bend_range_semitones,
                        bend_alpha=bend_alpha,
                        bend_fixed_base=bend_fixed_base,
                    )
                    name = _sanitize_name(res.instrument.name)
                    res.instrument.name = name
                    results.append((name, res.instrument, res.tempo))
                    total_time += time.perf_counter() - start

            tempo: float | None = None
            summary_line: str | None = None
            tempos = [float(t) for _, _, t in results if t is not None and math.isfinite(float(t))]
            if tempo_lock != "none":
                orig_mode = tempo_lock
                candidates = [(n, float(t)) for n, _, t in results if t is not None and math.isfinite(float(t))]
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
                            if tempo_fold_halves:
                                candidates = [(n, _fold_to_ref(t, anchor_bpm)) for n, t in candidates]
                            tempo = float(anchor_bpm)
                            summary_line = (
                                f"Tempo-lock(anchor): {song_name} → BPM={tempo:.1f} "
                                f"(pattern='{tempo_anchor_pattern}', fold_halves={tempo_fold_halves}, "
                                f"candidates={cand_count}, anchor='{anchor_name}')"
                            )
                        else:
                            tempo_lock = "median"
                    if tempo_lock == "median":
                        vals = [t for _, t in candidates]
                        if not vals:
                            tempo = 120.0
                            logger.warning(
                                "Tempo-lock(%s): %s has no valid tempo estimates (%d stems); using 120 BPM",
                                orig_mode,
                                song_name,
                                len(results),
                            )
                        else:
                            tempo = (
                                _folded_median(vals)
                                if tempo_fold_halves
                                else statistics.median(vals)
                            )
                        summary_line = (
                            "Tempo-lock(median): "
                            f"{song_name} → BPM={tempo:.1f} "
                            f"(fold_halves={tempo_fold_halves}, candidates={cand_count})"
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

            pm = (
                pretty_midi.PrettyMIDI(initial_tempo=float(tempo))
                if tempo is not None
                else pretty_midi.PrettyMIDI()
            )
            for name, inst, _ in results:
                _emit_pitch_bend_range(inst, bend_range_semitones)
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
                                min_dur=min_dur,
                                auto_tempo=auto_tempo,
                                enable_bend=enable_bend,
                                bend_range_semitones=bend_range_semitones,
                                bend_alpha=bend_alpha,
                                bend_fixed_base=bend_fixed_base,
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
                        min_dur=min_dur,
                        auto_tempo=auto_tempo,
                        enable_bend=enable_bend,
                        bend_range_semitones=bend_range_semitones,
                        bend_alpha=bend_alpha,
                        bend_fixed_base=bend_fixed_base,
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
                    summary_line = f"Tempo-lock(value): {song_name} → BPM={locked_tempo:.1f}"
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
                            if tempo_fold_halves:
                                candidates = [(n, _fold_to_ref(t, anchor_bpm)) for n, t in candidates]
                            locked_tempo = float(anchor_bpm)
                            summary_line = (
                                f"Tempo-lock(anchor): {song_name} → BPM={locked_tempo:.1f} "
                                f"(pattern='{tempo_anchor_pattern}', fold_halves={tempo_fold_halves}, "
                                f"candidates={cand_count}, anchor='{anchor_name}')"
                            )
                        else:
                            tempo_lock = "median"
                if tempo_lock == "median":
                    vals = [t for _, t in candidates]
                    if not vals:
                        locked_tempo = 120.0
                        logger.warning(
                            "Tempo-lock(%s): %s has no valid tempo estimates (%d stems); using 120 BPM",
                            orig_mode,
                            song_name,
                            len(results),
                        )
                    else:
                        locked_tempo = (
                            _folded_median(vals) if tempo_fold_halves else statistics.median(vals)
                        )
                    summary_line = (
                        "Tempo-lock(median): "
                        f"{song_name} → BPM={locked_tempo:.1f} "
                        f"(fold_halves={tempo_fold_halves}, candidates={cand_count})"
                    )

            for base, inst, tempo, midi_path in results:
                use_tempo = locked_tempo if locked_tempo is not None else tempo
                pm = (
                    pretty_midi.PrettyMIDI(initial_tempo=float(use_tempo))
                    if use_tempo is not None
                    else pretty_midi.PrettyMIDI()
                )
                _emit_pitch_bend_range(inst, bend_range_semitones)
                _coerce_note_times(inst)
                _coerce_controller_times(inst)
                pm.instruments.append(inst)
                pm.write(str(midi_path))
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
    args = parser.parse_args(argv)

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
    )


if __name__ == "__main__":
    main()
