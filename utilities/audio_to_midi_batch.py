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
import multiprocessing
import os
import re
import statistics
import time
import unicodedata
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import math
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


def _emit_pitch_bend_range(
    inst: pretty_midi.Instrument, bend_range_semitones: float, *, t: float = 0.0
) -> None:
    """Insert RPN 0,0 events to set pitch-bend range."""
    msb = int(max(0, min(127, math.floor(bend_range_semitones))))
    frac = bend_range_semitones - msb
    lsb = int(max(0, min(127, round(frac * 100))))
    cc = inst.control_changes
    cc.append(pretty_midi.ControlChange(number=101, value=0, time=float(t)))
    cc.append(pretty_midi.ControlChange(number=100, value=0, time=float(t)))
    cc.append(pretty_midi.ControlChange(number=6, value=msb, time=float(t)))
    cc.append(pretty_midi.ControlChange(number=38, value=lsb, time=float(t)))


def _generate_ccs(
    audio: "np.ndarray | list[float]",
    sr: int,
    inst: pretty_midi.Instrument,
    stem_name: str,
    *,
    cc_strategy: str,
    cc11_smoothing_ms: int,
    cc64_threshold: float,
    cc64_instruments: list[str],
    cc11_min_dt_ms: int,
    cc11_min_delta: int,
) -> None:
    """Populate ``inst`` with CC11/CC64 events derived from ``audio``."""
    if np is None or not len(audio):
        return
    hop = max(1, int(sr * 0.01))
    env = None
    if librosa is not None:
        try:
            env = librosa.feature.rms(y=audio, hop_length=hop)[0]
        except Exception:  # pragma: no cover
            env = None
    if env is None:
        win = hop
        padded = np.pad(audio.astype(float), (win // 2, win // 2), mode="constant")
        env = np.sqrt(
            np.convolve(padded ** 2, np.ones(win) / win, mode="valid")[::hop]
        )
    if not env.size:
        return
    env = env - env.min()
    if env.max() > 0:
        env = env / env.max()
    times = np.arange(env.size) * hop / sr

    cc11_events = 0
    if cc_strategy in {"energy", "rms"}:
        alpha = hop / (sr * (cc11_smoothing_ms / 1000.0)) if cc11_smoothing_ms > 0 else 1.0
        alpha = max(0.0, min(1.0, float(alpha)))
        ema = 0.0
        values = []
        for e in env:
            ema = alpha * e + (1 - alpha) * ema
            values.append(ema)
        prev = -1
        last_t = -1.0
        min_dt = cc11_min_dt_ms / 1000.0
        for t, e in zip(times, values):
            val = int(round(e * 127))
            if prev != -1 and abs(val - prev) < cc11_min_delta:
                continue
            if last_t != -1.0 and (t - last_t) < min_dt:
                continue
            inst.control_changes.append(
                pretty_midi.ControlChange(number=11, value=val, time=float(t))
            )
            prev = val
            last_t = t
            cc11_events += 1

    cc64_events = 0
    if cc64_threshold is not None and any(
        k.strip().lower() in stem_name.lower() for k in cc64_instruments
    ):
        notes = sorted(inst.notes, key=lambda n: n.start)
        for a, b in zip(notes, notes[1:]):
            start = float(a.end)
            end = float(b.start)
            if end <= start:
                continue
            mask = (times >= start) & (times <= end)
            if not mask.any():
                continue
            if env[mask].mean() >= cc64_threshold:
                inst.control_changes.append(
                    pretty_midi.ControlChange(number=64, value=127, time=start)
                )
                inst.control_changes.append(
                    pretty_midi.ControlChange(
                        number=64, value=0, time=max(start, end - 0.01)
                    )
                )
                cc64_events += 2
    if cc11_events or cc64_events:
        logger.info(
            "strategy=%s events_cc11=%d events_cc64=%d smoothing=%dms",
            cc_strategy,
            cc11_events,
            cc64_events,
            cc11_smoothing_ms,
        )

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
    if np is None:
        raise RuntimeError("numpy is required for onset-only transcription")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
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
    cc_strategy: str = "none",
    cc11_smoothing_ms: int = 80,
    cc64_threshold: float = 0.6,
    cc64_instruments: list[str] | None = None,
    cc11_min_dt_ms: int = 30,
    cc11_min_delta: int = 3,
) -> StemResult:
    """Transcribe a monophonic WAV file into a MIDI instrument.

    Returns the instrument along with an estimated tempo (in BPM) when
    ``auto_tempo`` is enabled. Tempo estimation falls back to ``None`` if it
    cannot be computed."""

    if crepe is None or librosa is None:
        missing = " and ".join(
            dep for dep, mod in (("crepe", crepe), ("librosa", librosa)) if mod is None
        )
        logging.warning(
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
        result = _fallback_transcribe_stem(path, min_dur=min_dur, tempo=tempo)
        if audio is None:
            try:
                from scipy.io import wavfile

                sr, data = wavfile.read(path)
                audio = data.astype(float)
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
            except Exception:  # pragma: no cover - best effort
                audio = None
        if audio is not None:
            _generate_ccs(
                audio,
                sr,
                result.instrument,
                path.stem,
                cc_strategy=cc_strategy,
                cc11_smoothing_ms=cc11_smoothing_ms,
                cc64_threshold=cc64_threshold,
                cc64_instruments=cc64_instruments or ["piano", "ep", "keys"],
                cc11_min_dt_ms=cc11_min_dt_ms,
                cc11_min_delta=cc11_min_delta,
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
    _generate_ccs(
        audio,
        sr,
        inst,
        path.stem,
        cc_strategy=cc_strategy,
        cc11_smoothing_ms=cc11_smoothing_ms,
        cc64_threshold=cc64_threshold,
        cc64_instruments=cc64_instruments or ["piano", "ep", "keys"],
        cc11_min_dt_ms=cc11_min_dt_ms,
        cc11_min_delta=cc11_min_delta,
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
    enable_bend: bool = True,
    bend_range_semitones: float = 2.0,
    bend_alpha: float = 0.25,
    bend_fixed_base: bool = False,
    cc_strategy: str = "none",
    cc11_smoothing_ms: int = 80,
    cc64_threshold: float = 0.6,
    cc64_instruments: list[str] | None = None,
    cc11_min_dt_ms: int = 30,
    cc11_min_delta: int = 3,
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
    if resume and log_path.exists():
        try:
            log_data = json.loads(log_path.read_text())
        except Exception:
            logging.warning("Failed to read %s", log_path)

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
                logging.info("Skipping %s", out_song)
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
            tempos: list[float] = []
            insts: list[pretty_midi.Instrument] = []
            ex_kwargs = {"max_workers": jobs}
            if os.name != "nt":
                ex_kwargs["mp_context"] = multiprocessing.get_context("forkserver")
            if jobs > 1:
                with ProcessPoolExecutor(**ex_kwargs) as ex:
                    futures = []
                    for wav in wavs:
                        logging.info("Transcribing %s", wav)
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
                                cc_strategy=cc_strategy,
                                cc11_smoothing_ms=cc11_smoothing_ms,
                                cc64_threshold=cc64_threshold,
                                cc64_instruments=cc64_instruments,
                                cc11_min_dt_ms=cc11_min_dt_ms,
                                cc11_min_delta=cc11_min_delta,
                            ), start)
                        )
                    for fut, start in futures:
                        res = fut.result()
                        if res.tempo is not None:
                            tempos.append(res.tempo)
                        insts.append(res.instrument)
                        total_time += time.perf_counter() - start
            else:
                for wav in wavs:
                    logging.info("Transcribing %s", wav)
                    start = time.perf_counter()
                    res = _transcribe_stem(
                        wav,
                        min_dur=min_dur,
                        auto_tempo=auto_tempo,
                        enable_bend=enable_bend,
                        bend_range_semitones=bend_range_semitones,
                        bend_alpha=bend_alpha,
                        bend_fixed_base=bend_fixed_base,
                        cc_strategy=cc_strategy,
                        cc11_smoothing_ms=cc11_smoothing_ms,
                        cc64_threshold=cc64_threshold,
                        cc64_instruments=cc64_instruments,
                        cc11_min_dt_ms=cc11_min_dt_ms,
                        cc11_min_delta=cc11_min_delta,
                    )
                    if res.tempo is not None:
                        tempos.append(res.tempo)
                    insts.append(res.instrument)
                    total_time += time.perf_counter() - start

            tempo: float | None = None
            if tempo_strategy == "first":
                tempo = tempos[0] if tempos else None
            elif tempo_strategy == "median":
                if tempos:
                    spread = max(tempos) - min(tempos)
                    tempo = statistics.median(tempos)
                    if spread > 5:
                        logging.warning("Tempo spread %.1f BPM for %s", spread, song_name)
            elif tempo_strategy == "ignore":
                tempo = None

            pm = (
                pretty_midi.PrettyMIDI(initial_tempo=float(tempo))
                if tempo is not None
                else pretty_midi.PrettyMIDI()
            )
            for inst in insts:
                inst.name = _sanitize_name(inst.name)
                _emit_pitch_bend_range(inst, bend_range_semitones)
                _coerce_note_times(inst)
                _coerce_controller_times(inst)
                pm.instruments.append(inst)
            pm.write(str(out_song))
            converted = len(insts)
            if resume:
                log_data[song_name] = ["__MERGED__"]
                try:
                    log_path.write_text(json.dumps(log_data))
                except Exception:
                    logging.warning("Failed to update %s", log_path)
            logging.info("Wrote %s", out_song)
        else:
            processed = set(log_data.get(song_name, []))
            existing = {p.stem for p in out_song.glob("*.mid")}
            used_names: set[str] = set()
            tasks = []
            for wav in wavs:
                sanitized = _sanitize_name(wav.stem)
                if resume and not overwrite and sanitized in processed:
                    logging.info("Skipping %s", wav)
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

            if jobs > 1:
                ex_kwargs = {"max_workers": jobs}
                if os.name != "nt":
                    ex_kwargs["mp_context"] = multiprocessing.get_context("forkserver")
                with ProcessPoolExecutor(**ex_kwargs) as ex:
                    futures = {}
                    for wav, base, midi_path in tasks:
                        logging.info("Transcribing %s", wav)
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
                                cc_strategy=cc_strategy,
                                cc11_smoothing_ms=cc11_smoothing_ms,
                                cc64_threshold=cc64_threshold,
                                cc64_instruments=cc64_instruments,
                                cc11_min_dt_ms=cc11_min_dt_ms,
                                cc11_min_delta=cc11_min_delta,
                            )
                        ] = (base, midi_path, start)
                    for fut in as_completed(futures):
                        base, midi_path, start = futures[fut]
                        res = fut.result()
                        inst = res.instrument
                        tempo = res.tempo
                        inst.name = base
                        pm = (
                            pretty_midi.PrettyMIDI(initial_tempo=float(tempo))
                            if tempo is not None
                            else pretty_midi.PrettyMIDI()
                        )
                        _emit_pitch_bend_range(inst, bend_range_semitones)
                        _coerce_note_times(inst)
                        _coerce_controller_times(inst)
                        pm.instruments.append(inst)
                        pm.write(str(midi_path))
                        total_time += time.perf_counter() - start
                        converted += 1
                        processed.add(base)
                        if resume:
                            log_data[song_name] = sorted(processed)
                            try:
                                log_path.write_text(json.dumps(log_data))
                            except Exception:
                                logging.warning("Failed to update %s", log_path)
                        logging.info("Wrote %s", midi_path)
            else:
                for wav, base, midi_path in tasks:
                    logging.info("Transcribing %s", wav)
                    start = time.perf_counter()
                    res = _transcribe_stem(
                        wav,
                        min_dur=min_dur,
                        auto_tempo=auto_tempo,
                        enable_bend=enable_bend,
                        bend_range_semitones=bend_range_semitones,
                        bend_alpha=bend_alpha,
                        bend_fixed_base=bend_fixed_base,
                        cc_strategy=cc_strategy,
                        cc11_smoothing_ms=cc11_smoothing_ms,
                        cc64_threshold=cc64_threshold,
                        cc64_instruments=cc64_instruments,
                        cc11_min_dt_ms=cc11_min_dt_ms,
                        cc11_min_delta=cc11_min_delta,
                    )
                    inst = res.instrument
                    tempo = res.tempo
                    total_time += time.perf_counter() - start
                    inst.name = base
                    pm = (
                        pretty_midi.PrettyMIDI(initial_tempo=float(tempo))
                        if tempo is not None
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
                            logging.warning("Failed to update %s", log_path)
                    logging.info("Wrote %s", midi_path)

        logging.info(
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
        "--cc-strategy",
        choices=["none", "energy", "rms"],
        default="none",
        help="Derive CC11 from audio energy using the chosen strategy",
    )
    parser.add_argument(
        "--cc11-smoothing-ms",
        type=int,
        default=80,
        help="Smoothing window for CC11 envelope in milliseconds",
    )
    parser.add_argument(
        "--cc64-threshold",
        type=float,
        default=0.6,
        help="Energy threshold for heuristic sustain pedal insertion",
    )
    parser.add_argument(
        "--cc64-instruments",
        type=str,
        default="piano,ep,keys",
        help="Comma-separated substrings enabling sustain heuristic",
    )
    parser.add_argument(
        "--cc11-min-dt-ms",
        type=int,
        default=30,
        help="Minimum interval between successive CC11 events",
    )
    parser.add_argument(
        "--cc11-min-delta",
        type=int,
        default=3,
        help="Minimum value change between CC11 events",
    )
    parser.add_argument(
        "--controls-post-bend",
        choices=["skip", "add", "replace"],
        default="skip",
        help="How to merge synthesized controls after existing pitch bends",
    )
    args = parser.parse_args(argv)

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
        enable_bend=args.enable_bend,
        bend_range_semitones=args.bend_range_semitones,
        bend_alpha=args.bend_alpha,
        bend_fixed_base=args.bend_fixed_base,
        cc_strategy=args.cc_strategy,
        cc11_smoothing_ms=args.cc11_smoothing_ms,
        cc64_threshold=args.cc64_threshold,
        cc64_instruments=[s.strip() for s in args.cc64_instruments.split(",") if s.strip()],
        cc11_min_dt_ms=args.cc11_min_dt_ms,
        cc11_min_delta=args.cc11_min_delta,
        controls_post_bend=args.controls_post_bend,
    )


if __name__ == "__main__":
    main()
