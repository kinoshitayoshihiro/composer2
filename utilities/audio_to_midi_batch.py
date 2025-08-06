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
    [--quiet]
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
``--safe-dirnames`` sanitizes song directory names for the output tree. ``--quiet``
disables progress bars. The converter logs each WAV file as it is transcribed
and reports when the corresponding MIDI file is written. Standard ``logging``
configuration can be used to silence or redirect this output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import multiprocessing
import re
import time
import unicodedata
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pretty_midi
from tqdm.auto import tqdm

LOG_VERSION = 1

try:  # Optional heavy deps
    import crepe  # type: ignore
except Exception:  # pragma: no cover - handled by fallback
    crepe = None  # type: ignore

try:  # Optional heavy deps
    import librosa  # type: ignore
except Exception:  # pragma: no cover - handled by fallback
    librosa = None  # type: ignore


def _sanitize_name(name: str) -> str:
    """Return a best-effort ASCII-only representation of ``name``."""

    normalized = unicodedata.normalize("NFKD", name)
    ascii_name = normalized.encode("ascii", "ignore").decode("ascii")
    # Replace whitespace with underscores and drop other invalid chars
    ascii_name = re.sub(r"\s+", "_", ascii_name)
    ascii_name = re.sub(r"[^\w-]", "", ascii_name)
    return ascii_name or "track"


def _fallback_transcribe_stem(path: Path, *, min_dur: float) -> pretty_midi.Instrument:
    """Simpler onset-only transcription used when CREPE/librosa are unavailable."""

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
        return inst

    import numpy as np
    from scipy.io import wavfile

    sr, audio = wavfile.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(float)
    envelope = np.abs(audio)
    if not envelope.size:
        return inst
    thresh = 0.5 * envelope.max()
    onset_idxs = np.where((envelope[1:] >= thresh) & (envelope[:-1] < thresh))[0]
    for idx in onset_idxs:
        t = idx / sr
        inst.notes.append(
            pretty_midi.Note(
                velocity=100,
                pitch=36,  # Kick drum placeholder until drum mapping is improved
                start=float(t),
                end=float(t + min_dur),
            )
        )
    return inst


def _transcribe_stem(
    path: Path,
    *,
    step_size: int = 10,
    conf_threshold: float = 0.5,
    min_dur: float = 0.05,
) -> pretty_midi.Instrument:
    """Transcribe a monophonic WAV file into a MIDI instrument."""

    if crepe is None or librosa is None:
        missing = " and ".join(
            dep for dep, mod in (("crepe", crepe), ("librosa", librosa)) if mod is None
        )
        logging.warning(
            "Missing %s; falling back to onset-only transcription, "
            "transcription quality may degrade",
            missing,
        )
        return _fallback_transcribe_stem(path, min_dur=min_dur)

    audio, sr = librosa.load(path, sr=16000, mono=True)
    time, freq, conf, _ = crepe.predict(
        audio, sr, step_size=step_size, model_capacity="full", verbose=0
    )

    inst = pretty_midi.Instrument(program=0, name=path.stem)
    pitch: int | None = None
    start: float = 0.0

    for t, f, c in zip(time, freq, conf):
        if c < conf_threshold:
            if pitch is not None and t - start >= min_dur:
                inst.notes.append(
                    pretty_midi.Note(
                        velocity=100, pitch=pitch, start=start, end=float(t)
                    )
                )
            pitch = None
            continue

        p = int(round(pretty_midi.hz_to_note_number(f)))
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

    if pitch is not None and time.size:
        end = float(time[-1])
        if end - start >= min_dur:
            inst.notes.append(
                pretty_midi.Note(velocity=100, pitch=pitch, start=start, end=end)
            )

    return inst


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


def _write_log(path: Path, data: dict[str, list[str] | int]) -> None:
    data["__version__"] = LOG_VERSION
    try:
        path.write_text(json.dumps(data))
    except Exception:
        logging.warning("Failed to update %s", path)


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
    quiet: bool = False,
) -> None:
    """Convert a directory of stems into individual MIDI files.

    If ``merge`` is ``True``, all stems for a song are merged into a single
    multi-track MIDI file, emulating the legacy behaviour.
    """

    exts = [e.strip().lstrip(".") for e in ext.split(",") if e.strip()]
    dst.mkdir(parents=True, exist_ok=True)

    log_path = dst / "conversion_log.json"
    log_data: dict[str, list[str] | int] = {}
    if resume and log_path.exists():
        try:
            log_data = json.loads(log_path.read_text())
        except Exception:
            logging.warning("Failed to read %s", log_path)
        else:
            version = log_data.get("__version__", 0)  # type: ignore[assignment]
            if version != LOG_VERSION:
                logging.warning("Ignoring %s with version %s", log_path, version)
                log_data = {"__version__": LOG_VERSION}

    song_dirs = _iter_song_dirs(src, exts)
    for song_dir in tqdm(song_dirs, desc="Songs", disable=quiet):
        song_name_raw = song_dir.name
        song_name = _sanitize_name(song_name_raw) if safe_dirnames else song_name_raw
        if safe_dirnames:
            sha = hashlib.sha1(song_name_raw.encode()).hexdigest()[:8]
            base = song_name
            hashed = f"{base}__{sha}"
            base_path = dst / (f"{base}.mid" if merge else base)
            hash_path = dst / (f"{hashed}.mid" if merge else hashed)
            if hash_path.exists() or (base_path.exists() and base not in log_data):
                song_name = hashed

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
            pm = pretty_midi.PrettyMIDI()
            if jobs > 1:
                multiprocessing.set_start_method("forkserver", force=True)
                with ProcessPoolExecutor(max_workers=jobs) as ex:
                    futures = []
                    for wav in wavs:
                        logging.info("Transcribing %s", wav)
                        start = time.perf_counter()
                        futures.append(
                            (ex.submit(_transcribe_stem, wav, min_dur=min_dur), start)
                        )
                    stem_bar = tqdm(
                        total=len(futures), desc=song_name, disable=quiet, leave=False
                    )
                    for fut, start in futures:
                        inst = fut.result()
                        total_time += time.perf_counter() - start
                        pm.instruments.append(inst)
                        stem_bar.update(1)
                    stem_bar.close()
            else:
                stem_bar = tqdm(wavs, desc=song_name, disable=quiet, leave=False)
                for wav in stem_bar:
                    logging.info("Transcribing %s", wav)
                    start = time.perf_counter()
                    pm.instruments.append(_transcribe_stem(wav, min_dur=min_dur))
                    total_time += time.perf_counter() - start
                stem_bar.close()
            for inst in pm.instruments:
                inst.name = _sanitize_name(inst.name)
            pm.write(str(out_song))
            converted = len(pm.instruments)
            if resume:
                log_data[song_name] = ["__MERGED__"]
                _write_log(log_path, log_data)
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
                multiprocessing.set_start_method("forkserver", force=True)
                stem_bar = tqdm(
                    total=len(tasks), desc=song_name, disable=quiet, leave=False
                )
                with ProcessPoolExecutor(max_workers=jobs) as ex:
                    futures = {}
                    for wav, base, midi_path in tasks:
                        logging.info("Transcribing %s", wav)
                        start = time.perf_counter()
                        futures[ex.submit(_transcribe_stem, wav, min_dur=min_dur)] = (
                            base,
                            midi_path,
                            start,
                        )
                    for fut in as_completed(futures):
                        base, midi_path, start = futures[fut]
                        inst = fut.result()
                        inst.name = base
                        pm = pretty_midi.PrettyMIDI()
                        pm.instruments.append(inst)
                        pm.write(str(midi_path))
                        total_time += time.perf_counter() - start
                        converted += 1
                        processed.add(base)
                        if resume:
                            log_data[song_name] = sorted(processed)
                            _write_log(log_path, log_data)
                        logging.info("Wrote %s", midi_path)
                        stem_bar.update(1)
                stem_bar.close()
            else:
                stem_bar = tqdm(tasks, desc=song_name, disable=quiet, leave=False)
                for wav, base, midi_path in stem_bar:
                    logging.info("Transcribing %s", wav)
                    start = time.perf_counter()
                    inst = _transcribe_stem(wav, min_dur=min_dur)
                    total_time += time.perf_counter() - start
                    inst.name = base
                    pm = pretty_midi.PrettyMIDI()
                    pm.instruments.append(inst)
                    pm.write(str(midi_path))
                    converted += 1
                    processed.add(base)
                    if resume:
                        log_data[song_name] = sorted(processed)
                        _write_log(log_path, log_data)
                    logging.info("Wrote %s", midi_path)
                stem_bar.close()

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
        "--quiet",
        action="store_true",
        help="Disable progress bars",
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
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()
