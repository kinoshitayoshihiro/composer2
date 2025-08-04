"""Batch convert stem WAV files to multi-track MIDI.

This utility walks a directory of audio stems and converts each group of
tracks into a single multi-track MIDI file. Each input WAV is transcribed
using `crepe` for pitch detection and `pretty_midi` for MIDI generation. The
stem's filename (sans extension) is preserved as the name of the corresponding
MIDI track.

Usage
-----
```
python -m utilities.audio_to_midi_batch src_dir dst_dir [--jobs N] [--ext EXT[,EXT...]] [--min-dur SEC] [--resume]
```

`src_dir` should contain sub-directories, one per song, each holding WAV
stems. If `src_dir` itself contains WAV files, they are treated as a single
song. The resulting MIDI files are written to `dst_dir` with the directory
name as the file name.

Passing ``--resume`` skips songs whose output MIDI files already exist. When
resuming, completed conversions are logged to ``conversion_log.json`` in the
destination directory for incremental processing. The converter logs each WAV
file as it is transcribed and reports when the final MIDI file is written.
Standard ``logging`` configuration can be used to silence or redirect this
output.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import unicodedata
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pretty_midi

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


def convert_directory(
    src: Path,
    dst: Path,
    *,
    ext: str = "wav",
    jobs: int = 1,
    min_dur: float = 0.05,
    resume: bool = False,
) -> None:
    """Convert a directory of stems into multi-track MIDI files."""

    exts = [e.strip().lstrip(".") for e in ext.split(",") if e.strip()]
    dst.mkdir(parents=True, exist_ok=True)

    log_path = dst / "conversion_log.json"
    processed: set[str] = set()
    if resume and log_path.exists():
        try:
            processed = set(json.loads(log_path.read_text()))
        except Exception:
            logging.warning("Failed to read %s", log_path)

    for song_dir in _iter_song_dirs(src, exts):
        out_path = dst / f"{song_dir.name}.mid"
        if resume and (out_path.exists() or song_dir.name in processed):
            logging.info("Skipping %s", out_path)
            continue
        if len(exts) == 1:
            wavs = sorted(song_dir.glob(f"*.{exts[0]}"))
        else:
            wavs = []
            for e in exts:
                wavs.extend(song_dir.glob(f"*.{e}"))
            wavs.sort()
        if not wavs:
            continue

        pm = pretty_midi.PrettyMIDI()
        if jobs > 1:
            with ProcessPoolExecutor(max_workers=jobs) as ex:
                futures = []
                for wav in wavs:
                    logging.info("Transcribing %s", wav)
                    futures.append(ex.submit(_transcribe_stem, wav, min_dur=min_dur))
                for future in futures:
                    pm.instruments.append(future.result())
        else:
            for wav in wavs:
                logging.info("Transcribing %s", wav)
                pm.instruments.append(_transcribe_stem(wav, min_dur=min_dur))
        # Ensure MIDI track names contain only ASCII to avoid encoding errors
        for inst in pm.instruments:
            inst.name = _sanitize_name(inst.name)

        pm.write(str(out_path))
        logging.info("Wrote %s", out_path)
        if resume:
            processed.add(song_dir.name)
            try:
                log_path.write_text(json.dumps(sorted(processed)))
            except Exception:
                logging.warning("Failed to update %s", log_path)


def main(argv: list[str] | None = None) -> None:
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Batch audio-to-MIDI converter")
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
        help="Skip songs with existing MIDI files and log progress",
    )
    args = parser.parse_args(argv)

    convert_directory(
        Path(args.src_dir),
        Path(args.dst_dir),
        ext=args.ext,
        jobs=args.jobs,
        min_dur=args.min_dur,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
