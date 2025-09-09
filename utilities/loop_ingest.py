from __future__ import annotations

import concurrent.futures
import glob
import logging
import pickle
import warnings
from collections.abc import Sequence
from pathlib import Path
from statistics import mean
from typing import Any, TypedDict

import click
import mido
import yaml

from utilities.pretty_midi_safe import new_pm as PrettyMIDI

from .drum_map_registry import GM_DRUM_MAP
from .midi_utils import safe_end_time
from .types import Intensity

logger = logging.getLogger(__name__)

try:  # optional dependency
    import librosa

    HAVE_LIBROSA = True
except Exception:  # pragma: no cover - optional dependency missing
    librosa = None  # type: ignore
    HAVE_LIBROSA = False

try:  # older SciPy compatibility
    from scipy import signal

    if not hasattr(signal, "hann") and hasattr(signal, "windows"):
        signal.hann = signal.windows.hann  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - optional dependency
    pass

Token = tuple[int, str, int, int]


class LoopEntry(TypedDict, total=False):
    file: str
    tokens: list[Token]
    tempo_bpm: float
    bar_beats: int
    section: str
    heat_bin: int
    intensity: Intensity


_PITCH_TO_LABEL = {val[1]: key for key, val in GM_DRUM_MAP.items()}


def load_meta(path: Path) -> dict[str, Any]:
    """Load auxiliary metadata next to *path*.

    Looks for ``<name>.meta.yaml`` or ``<name>.meta.yml`` and returns the
    parsed dictionary if available. Unknown formats are ignored.
    """
    for suf in (".meta.yaml", ".meta.yml"):
        meta_path = path.with_suffix(suf)
        if meta_path.is_file():
            try:
                with meta_path.open("r", encoding="utf-8") as fh:
                    data = yaml.safe_load(fh) or {}
                if isinstance(data, dict):
                    return {str(k): data[k] for k in data}
            except Exception as exc:  # pragma: no cover - malformed yaml
                logger.warning("Failed to load meta for %s: %s", path, exc)
            break
    return {}


def _load_pretty_midi(path: Path) -> PrettyMIDI | None:
    """Load a MIDI file with timeout and size guard."""

    try:
        mf = mido.MidiFile(str(path))
        if mf.length > 600 or len(mf.tracks) > 32:
            logger.warning(
                "Skipping %s (too large: %.1fs, %d tracks)",
                path,
                mf.length,
                len(mf.tracks),
            )
            return None
    except Exception as exc:  # pragma: no cover - invalid MIDI
        logger.warning("Failed to inspect %s: %s", path, exc)
        return None

    def _load() -> PrettyMIDI:
        return PrettyMIDI(str(path))

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_load)
        try:
            return fut.result(timeout=10)
        except concurrent.futures.TimeoutError:
            logger.warning("Timed out loading %s", path)
        except Exception as exc:
            logger.warning("Failed to load %s: %s", path, exc)
    return None


def _quantize(
    beat: float, bar_beats: int, resolution: int, ppq: int
) -> tuple[int, int]:
    raw_step = beat * resolution / bar_beats
    step = int(round(raw_step))
    clamped = max(0, min(resolution - 1, step))
    if step != clamped and not (step == resolution and beat >= bar_beats):
        warnings.warn(
            f"quantised step {step} outside 0..{resolution - 1}; clamped",
            RuntimeWarning,
        )
    step = clamped
    q_pos = step * bar_beats / resolution
    micro = int(round((beat - q_pos) * ppq))
    return step, micro


_PERC_PITCH_TO_LABEL = {
    39: "conga_cl",
    40: "conga_op",
    70: "shaker",
    71: "shaker",
    72: "tamb",
    73: "tamb",
    74: "clap",
    75: "clap",
    76: "cowbell",
    77: "cowbell",
}


def _scan_midi(
    path: Path, resolution: int, ppq: int, *, part: str = "drums"
) -> LoopEntry | None:
    pm = _load_pretty_midi(path)
    if pm is None:
        return None
    times, tempi = pm.get_tempo_changes()
    bpm = float(tempi[0]) if len(tempi) > 0 else 120.0
    if bpm <= 0:
        logger.warning(
            "Non-positive tempo %.2f detected in %s; using default 120 BPM.",
            bpm,
            path,
        )
        bpm = 120.0
    sec_per_beat = 60.0 / bpm
    bar_beats = max(1, int(round(safe_end_time(pm) / sec_per_beat)))
    tokens: list[Token] = []
    for inst in pm.instruments:
        if not inst.is_drum:
            continue
        for note in inst.notes:
            beat = note.start / sec_per_beat
            step, micro = _quantize(beat, bar_beats, resolution, ppq)
            if part == "perc":
                label = _PERC_PITCH_TO_LABEL.get(note.pitch)
                if not label:
                    continue
            else:
                label = _PITCH_TO_LABEL.get(note.pitch, str(note.pitch))
            tokens.append((step, label, note.velocity, micro))
    return {
        "file": path.name,
        "tokens": tokens,
        "tempo_bpm": bpm,
        "bar_beats": bar_beats,
    }


def _scan_wav(
    path: Path, resolution: int, ppq: int, *, part: str = "drums"
) -> LoopEntry:
    import numpy as np
    import soundfile as sf

    y, sr = sf.read(path, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)

    bpm = 120.0
    sec_per_beat = 60.0 / bpm
    bar_beats = max(1, int(round(len(y) / sr / sec_per_beat)))

    threshold = 0.2
    min_gap = int(0.1 * sr)
    onset_samples = []
    last_onset = -min_gap
    for i, val in enumerate(y):
        if val > threshold and i - last_onset >= min_gap:
            onset_samples.append(i)
            last_onset = i

    tokens: list[Token] = []
    for idx, s in enumerate(onset_samples):
        end = onset_samples[idx + 1] if idx + 1 < len(onset_samples) else len(y)
        seg = y[s:end]
        beat = (s / sr) / sec_per_beat
        step, micro = _quantize(beat, bar_beats, resolution, ppq)
        label = "perc"
        if part == "perc":
            import numpy as np

            fft = np.abs(np.fft.rfft(seg))
            freqs = np.fft.rfftfreq(len(seg), 1 / sr)
            f = freqs[np.argmax(fft)] if len(fft) else 0.0
            dur_ms = len(seg) / sr * 1000
            if f < 120:
                label = "conga_cl"
            elif f < 300:
                label = "conga_op"
            elif f > 3000 and dur_ms < 120:
                label = "shaker"
            else:
                label = "perc_other"
        tokens.append((step, label, 100, micro))

    return {
        "file": path.name,
        "tokens": tokens,
        "tempo_bpm": bpm,
        "bar_beats": bar_beats,
    }


def _scan_file(
    path: Path, resolution: int, ppq: int, *, part: str = "drums"
) -> LoopEntry | None:
    suf = path.suffix.lower()
    entry: LoopEntry | None = None
    if suf in {".mid", ".midi"}:
        entry = _scan_midi(path, resolution, ppq, part=part)
    elif suf == ".wav":
        try:
            entry = _scan_wav(path, resolution, ppq, part=part)
        except TypeError:
            entry = _scan_wav(path, resolution, ppq)
        except RuntimeError:
            return None
    if entry is not None:
        entry["file"] = path.name
        meta = load_meta(path)
        if meta:
            entry["aux"] = meta  # type: ignore[typeddict-item]
    return entry


def scan_loops(
    loop_dir: Path,
    exts: Sequence[str] | None = None,
    resolution: int = 16,
    ppq: int = 480,
    progress: bool = False,
    *,
    part: str = "drums",
) -> list[LoopEntry]:
    """Return token sequences for all loops in ``loop_dir``.

    Args:
        loop_dir: Directory containing loops.
        exts: Extensions to load.
        resolution: Quantisation steps per bar.
        ppq: Pulses per quarter note for micro offsets.
        progress: Show a progress bar when more than 100 files are scanned.
    """

    extset = {
        e.lower().lstrip(".").replace("midi", "mid") for e in (exts or ["mid", "wav"])
    }
    data: list[LoopEntry] = []
    files = [
        p
        for p in sorted(loop_dir.iterdir())
        if p.suffix.lower().lstrip(".") in extset
        or (p.suffix.lower().lstrip(".") == "midi" and "mid" in extset)
    ]
    iterator: Sequence[Path] = files
    bar = None
    if progress and len(files) > 100:
        try:
            from tqdm import tqdm  # type: ignore

            bar = tqdm(files, unit="file")
            iterator = bar  # type: ignore[assignment]
        except Exception:
            pass
    for path in iterator:
        suf = path.suffix.lower().lstrip(".")
        if suf == "midi":
            suf = "mid"
        if suf not in extset:
            continue
        entry = _scan_file(path, resolution, ppq, part=part)
        if entry:
            data.append(entry)
    if bar is not None:
        bar.close()
    return data


def save_cache(
    data: Sequence[LoopEntry], out_path: Path, *, ppq: int, resolution: int
) -> None:
    """Serialize ``data`` to ``out_path`` using pickle."""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as fh:
        pickle.dump({"ppq": ppq, "resolution": resolution, "data": list(data)}, fh)


def load_cache(path: Path) -> list[LoopEntry]:
    obj: object
    with path.open("rb") as fh:
        obj = pickle.load(fh)
    if not isinstance(obj, dict) or not {
        "ppq",
        "resolution",
        "data",
    }.issubset(obj.keys()):
        raise ValueError("invalid loop cache")
    return list(obj["data"])


def collate_multi_part(
    sequences: Sequence[dict[str, Sequence[int]]],
    parts: Sequence[str],
    pad: int = 0,
) -> dict[str, list[list[int]]]:
    """Align multi-part integer sequences with padding."""

    max_len = 0
    for seq in sequences:
        for p in parts:
            max_len = max(max_len, len(seq.get(p, [])))
    result = {p: [] for p in parts}
    for seq in sequences:
        for p in parts:
            part_seq = list(seq.get(p, []))
            if len(part_seq) < max_len:
                part_seq += [pad] * (max_len - len(part_seq))
            result[p].append(part_seq)
    return result


# --------------------------- CLI -------------------------------------------


@click.group()
def cli() -> None:
    """Loop ingestion utilities."""


@cli.command()
@click.argument("paths", nargs=-1, type=str)
@click.option("--ext", default="mid,wav", help="Comma separated extensions")
@click.option("--out", "out_path", type=Path, default=Path("cache.pkl"))
@click.option("--resolution", default=16, type=int)
@click.option("--ppq", default=480, type=int)
@click.option("--progress/--no-progress", default=True)
@click.option("--auto-aux/--no-auto-aux", default=False, help="Infer aux metadata")
@click.option("--part", type=str, default="drums", help="drums or perc")
def scan(
    paths: tuple[str, ...],
    ext: str,
    out_path: Path,
    resolution: int,
    ppq: int,
    progress: bool,
    auto_aux: bool,
    part: str,
) -> None:
    """Scan loops and save a token cache.

    If ``auto_aux`` is true missing ``section``, ``heat_bin`` and ``intensity``
    are inferred for each loop. The mean velocity determines the intensity bucket:
    ``<=60`` -> ``low``, ``61-100`` -> ``mid``, ``>100`` -> ``high``.
    The heat bin is the step with the highest hit count modulo ``16``.
    """

    extset = {
        e.strip().lower().replace("midi", "mid") for e in ext.split(",") if e.strip()
    }
    file_list: list[Path] = []
    if not paths:
        paths = (".",)
    for pat in paths:
        for match in glob.glob(pat):
            p = Path(match)
            if p.is_dir():
                for f in p.iterdir():
                    suf = f.suffix.lower().lstrip(".")
                    if suf == "midi":
                        suf = "mid"
                    if suf in extset:
                        file_list.append(f)
            elif p.is_file():
                suf = p.suffix.lower().lstrip(".")
                if suf == "midi":
                    suf = "mid"
                if suf in extset:
                    file_list.append(p)
    if not HAVE_LIBROSA and any(f.suffix.lower() == ".wav" for f in file_list):
        click.echo(
            'WAV files detected but skipped due to missing dependency "librosa". '
            "Install it with pip install librosa."
        )
    iterator = file_list
    bar = None
    if progress:
        try:
            from tqdm import tqdm  # type: ignore

            bar = tqdm(file_list, unit="file")
            iterator = bar  # type: ignore[assignment]
        except Exception:
            pass
    data: list[LoopEntry] = []
    skipped = 0
    warned_librosa = False
    for f in iterator:
        entry = _scan_file(f, resolution, ppq, part=part)
        if entry:
            if auto_aux:
                entry.setdefault("section", f.parent.name or "unknown")
                vel_mean = mean(t[2] for t in entry["tokens"]) if entry["tokens"] else 0
                entry["intensity"] = (
                    "low" if vel_mean <= 60 else "mid" if vel_mean <= 100 else "high"
                )
                entry["heat_bin"] = len(entry["tokens"]) % 16
            data.append(entry)
        elif f.suffix.lower() == ".wav" and not HAVE_LIBROSA:
            if not warned_librosa:
                click.echo(
                    'WAV files detected but skipped due to missing dependency "librosa". '
                    "Install it with pip install librosa."
                )
                warned_librosa = True
            skipped += 1
    if bar is not None:
        bar.close()
    save_cache(data, out_path, ppq=ppq, resolution=resolution)
    total = sum(len(d["tokens"]) for d in data)
    msg = f"{len(data)} files, {total} tokens"
    if skipped:
        msg += f" | WAV skipped (librosa missing): {skipped}"
    click.echo(msg)


@cli.command()
@click.argument("cache", type=Path)
def info(cache: Path) -> None:
    """Print summary statistics for a loop cache."""

    data = load_cache(cache)
    total = sum(len(d["tokens"]) for d in data)
    instruments = sorted({t[1] for d in data for t in d["tokens"]})
    tempos = [d["tempo_bpm"] for d in data]
    click.echo(f"files: {len(data)}")
    click.echo(f"total tokens: {total}")
    click.echo(f"instruments: {', '.join(instruments)}")
    click.echo(
        f"tempo BPM min/mean/max: {min(tempos):.1f}/{mean(tempos):.1f}/{max(tempos):.1f}"
    )


__all__ = [
    "LoopEntry",
    "load_meta",
    "scan_loops",
    "save_cache",
    "load_cache",
    "cli",
    "scan",
    "info",
    "main",
]


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover - CLI entry
    cli.main(args=list(argv) if argv is not None else None, standalone_mode=False)


if __name__ == "__main__":  # pragma: no cover
    main()
