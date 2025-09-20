"""Extract tempo rubato curves from MIDI performances.

The resulting DataFrame/CSV has columns:
    track_id: MIDI track index used for each beat
    beat:     zero-based beat index (``np.int32``)
    tempo_factor: (bpm_perf / bpm_score) - 1
    time_sec: absolute time of each beat in seconds

Use :func:`configure_logging` to set up a module logger. The CLI supports
``--overwrite`` to replace an existing CSV (exit code 1 on refusal),
``--log-level`` to adjust verbosity, ``--poly`` to set Savitzky-Golay order,
``--no-librosa`` to disable the librosa fallback, ``--track`` to pick the
primary track (negative=skip validation; abs used), ``--out`` to choose a custom
output path, ``--json`` to write JSON instead of CSV, and ``--quiet`` to silence
info logs. When ``return_df`` is ``False`` the function returns the path to the
written file; otherwise, a
``RubatoCurve`` (``pd.DataFrame``) is returned. Each row follows the
``RubatoRow`` schema.

Additional parameters:
    track_ids: sequence of track indexes validating input tracks
    use_librosa: enable librosa beat tracking fallback
    polyorder: polynomial order for Savitzky-Golay smoothing
    return_df: return DataFrame instead of writing CSV
    out_path: destination for CSV when ``return_df`` is ``False``
    as_json: write JSON instead of CSV when ``return_df`` is ``False``
    RubatoRow: schema describing each row in the resulting DataFrame

When smoothing is enabled, ``polyorder`` is clamped to ``window - 1`` if needed.
If ``beat_unit`` is ``8`` the beat index uses eighth-note resolution while the
timings in seconds are unchanged.

Example::

    from utilities.rubato_csv import extract_tempo_curve
    df = extract_tempo_curve("perf.mid", score_bpm=120)
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypedDict, overload

import numpy as np  # ruff: noqa: I001

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional dependency
    pd = None  # type: ignore

try:
    import pretty_midi  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pretty_midi = None  # type: ignore

if TYPE_CHECKING:  # pragma: no cover - type hints only
    from pretty_midi import PrettyMIDI

try:
    from scipy.signal import savgol_filter
except Exception:  # pragma: no cover - optional dependency
    savgol_filter = None  # type: ignore

try:
    import librosa  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    librosa = None  # type: ignore

__version__ = "0.6.0-rc10"

__all__: list[str] = sorted(
    [
        "__version__",
        "configure_logging",
        "extract_tempo_curve",
        "MidiLike",
        "RubatoCurve",
        "RubatoRow",
    ],
    key=str.lower,
)

logger = logging.getLogger(__name__)


def _records(
    track: int, beats: np.ndarray, tempo_factor: np.ndarray, beat_idx: np.ndarray
) -> list[RubatoRow]:
    return [
        {
            "track_id": int(track),
            "beat": int(b),
            "tempo_factor": float(tf),
            "time_sec": float(t),
        }
        for b, tf, t in zip(beat_idx, tempo_factor, beats)
    ]


class RubatoRow(TypedDict):
    track_id: int
    beat: int
    tempo_factor: float
    time_sec: float


RubatoCurve = pd.DataFrame

MidiLike = str | Path
if TYPE_CHECKING:  # pragma: no cover - type hints only
    MidiInput = MidiLike | PrettyMIDI
else:
    MidiInput = Any


def configure_logging(level: int = logging.INFO) -> None:
    """Configure basic logging for this module.

    The function is idempotent; repeated calls simply adjust the log level.
    """
    if logger.handlers:
        logger.setLevel(level)
        for h in logger.handlers:
            h.setLevel(level)
        return
    handler = logging.StreamHandler()
    handler.setLevel(level)
    logger.addHandler(handler)
    logger.setLevel(level)


if TYPE_CHECKING:
    from typing import TypedDict

    class _ECParams(TypedDict, total=False):
        score_bpm: float | None
        beat_unit: Literal[4, 8]
        track_ids: Sequence[int] | None
        smoothing_window_beats: int
        polyorder: int
        use_librosa: bool
        out_path: Path | None
        as_json: bool

    @overload
    def extract_tempo_curve(
        midi: MidiInput,
        *,
        return_df: Literal[True],
        **kwargs: _ECParams,
    ) -> RubatoCurve: ...

    @overload
    def extract_tempo_curve(
        midi: MidiInput,
        *,
        return_df: Literal[False],
        **kwargs: _ECParams,
    ) -> Path: ...


def _infer_score_bpm(pm: PrettyMIDI) -> float:
    _times, tempi = pm.get_tempo_changes()
    if len(tempi) == 0:
        raise ValueError("score_bpm is required if MIDI has no tempo map")
    return float(tempi[0])


def _choose_primary_track(pm: PrettyMIDI) -> int:
    for idx, inst in enumerate(pm.instruments):
        if not inst.is_drum and inst.program < 112:
            return idx
    return 0


def _detect_beats(
    pm: PrettyMIDI, bpm: float, *, use_librosa: bool = True
) -> np.ndarray:
    """Return beat times using PrettyMIDI or librosa fallback."""
    beats = pm.get_beats(initial_tempo=bpm)
    if len(beats):
        return beats
    if not use_librosa:
        raise RuntimeError("librosa disabled; cannot track beats")
    if librosa is None:
        raise RuntimeError("librosa is required for beat tracking")
    sr = 22050
    try:  # pragma: no cover - optional fallback
        audio = pm.fluidsynth()
        sr = 44100
    except Exception:  # pragma: no cover - fluidsynth missing
        audio = np.max(pm.get_piano_roll(fs=sr), axis=0).astype(np.float32)
    try:
        tempo, frames = librosa.beat.beat_track(y=audio, sr=sr)
        return librosa.frames_to_time(frames, sr=sr)
    except Exception:
        raise RuntimeError("librosa is required for beat tracking")


def extract_tempo_curve(
    midi: MidiInput,
    *,
    score_bpm: float | None = None,
    beat_unit: Literal[4, 8] = 4,
    track_ids: Sequence[int] | None = None,
    smoothing_window_beats: int = 0,
    polyorder: int = 1,
    use_librosa: bool = True,
    as_json: bool = False,
    return_df: bool = True,
    out_path: Path | None = None,
) -> pd.DataFrame | Path:
    """Extract tempo-rubato curve from MIDI and return DataFrame or CSV path.

    If ``return_df`` is ``False`` the DataFrame is written to ``out_path`` if
    provided, otherwise next to ``midi`` with suffix ``_rubato.csv`` or
    ``_rubato.json`` when ``as_json`` is ``True``. When smoothing,
    ``polyorder`` controls the Savitzky-Golay polynomial order. When
    ``track_ids`` is given, the first index labels each beat; invalid indices
    raise ``ValueError``.

    Parameters
    ----------
    as_json:
        Write JSON instead of CSV when ``return_df`` is ``False``.
    """

    if pretty_midi is None:
        raise ImportError("pretty_midi is required")

    start_time = time.perf_counter()

    midi_path_obj: Path | None = None
    if isinstance(midi, str | Path):
        midi_path_obj = Path(midi)
        pm = pretty_midi.PrettyMIDI(str(midi_path_obj))
    else:
        pm = midi

    bpm = score_bpm if score_bpm is not None else _infer_score_bpm(pm)
    beat_dur_score = 60.0 / float(bpm) * (beat_unit / 4)

    if track_ids is not None:
        track_ids = tuple(track_ids)

    beats = _detect_beats(pm, bpm, use_librosa=use_librosa)
    beat_index_factor = 2 if beat_unit == 8 else 1
    if len(beats) < 2:
        raise ValueError("at least two beats are required")

    delta = np.diff(beats)
    tempo_factor = delta / beat_dur_score - 1.0
    tempo_factor = np.concatenate([tempo_factor, tempo_factor[-1:]])

    if smoothing_window_beats > 0:
        if savgol_filter is None:
            logger.warning("SciPy not available; skipping smoothing")
        else:
            win = 2 * smoothing_window_beats + 1
            if win % 2 == 0:
                win += 1
            if win < 3 or win > len(tempo_factor):
                logger.warning(
                    "smoothing window %d invalid for %d beats; skipping",
                    win,
                    len(tempo_factor),
                )
            else:
                if polyorder >= win:
                    logger.warning(
                        "clamping polyorder %d to %d",
                        polyorder,
                        win - 1,
                        stacklevel=2,
                    )
                    polyorder = win - 1
                if polyorder <= 0:
                    logger.warning(
                        "polyorder %d invalid for window %d; skipping smoothing",
                        polyorder,
                        win,
                    )
                else:
                    logger.debug(
                        "smoothing beats=%d window=%d polyorder=%d",
                        smoothing_window_beats,
                        win,
                        polyorder,
                    )
                    tempo_factor = savgol_filter(
                        tempo_factor, win, polyorder, mode="nearest"
                    )

    if track_ids:
        first = track_ids[0]
        if first < 0:
            track = abs(first)
        else:
            invalid = [i for i in track_ids if i < 0 or i >= len(pm.instruments)]
            if invalid:
                raise ValueError(f"invalid track_ids: {tuple(invalid)}")
            track = first
    else:
        track = _choose_primary_track(pm)

    beat_idx = np.arange(len(beats), dtype=np.int32) * beat_index_factor
    if pd and hasattr(pd, "DataFrame"):
        df_full = pd.DataFrame(
             {
                 "track_id": np.full(beat_idx.shape, track, np.int32),
                 "beat": beat_idx,
                 "tempo_factor": tempo_factor.astype(np.float32),
                 "time_sec": beats.astype(np.float64),
             }
        )
        df = df_full.loc[:, ["track_id", "beat", "tempo_factor", "time_sec"]]
    else:  # pandas unavailable → list fallback
        df = _records(track, beats, tempo_factor, beat_idx)
    logger.debug("Extracted %d beats", len(df))
    elapsed = time.perf_counter() - start_time
    if logger.isEnabledFor(logging.INFO):
        logger.info("Extraction finished in %.3fs", elapsed)

    if return_df:
        return df

    if not (pd and hasattr(pd, "DataFrame")):
        raise ImportError("pandas is required when return_df is False")

    dest = out_path
    if dest is None:
        if midi_path_obj is None:
            raise ValueError(
                "out_path must be provided when midi is PrettyMIDI and return_df is False"
            )
        suffix = "_rubato.json" if as_json else "_rubato.csv"
        dest = midi_path_obj.with_name(midi_path_obj.stem + suffix)
    if as_json and dest.suffix != ".json":
        dest = dest.with_suffix(".json")
    elif not as_json and dest.suffix != ".csv":
        dest = dest.with_suffix(".csv")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if as_json:
        dest.write_text(df.to_json(orient="records", indent=2))
    else:
        df.to_csv(dest, index=False)
    return dest


def _cli() -> None:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Extract tempo rubato curve")
    parser.add_argument("midi", type=Path, help="MIDI file path")
    parser.add_argument("--score-bpm", type=float, default=None)
    parser.add_argument(
        "--beat-unit",
        type=int,
        choices=[4, 8],
        default=4,
        help="score beat unit (8 uses eighth-note beat index)",
    )
    parser.add_argument(
        "--track",
        type=int,
        default=None,
        help=("primary track id (negative=skip validation; abs used, e.g. --track -3)"),
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "-j",
        "--json",
        dest="as_json",
        action="store_true",
        help="write JSON instead of CSV",
    )
    parser.add_argument("--smoothing", type=int, default=0)
    parser.add_argument(
        "--poly",
        dest="polyorder",
        type=int,
        default=1,
        help="Savitzky-Golay polynomial order",
    )
    parser.add_argument("--overwrite", action="store_true", help="allow overwrite")
    parser.add_argument(
        "--no-librosa",
        dest="use_librosa",
        action="store_false",
        help="disable librosa beat tracking fallback",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="suppress info logs (sets WARNING level)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    parser.add_argument(
        "-V", "--version", action="store_true", help="print version and exit"
    )
    args = parser.parse_args()

    if args.version:
        print(__version__)
        return

    if args.quiet:
        configure_logging(logging.WARNING)
    else:
        configure_logging(getattr(logging, args.log_level))

    suffix = "_rubato.json" if args.as_json else "_rubato.csv"
    dest = args.out or Path(args.midi).with_name(Path(args.midi).stem + suffix)
    if dest.exists() and not args.overwrite:
        print(f"{dest} exists. Use --overwrite to replace.")
        sys.exit(1)
    out_path = extract_tempo_curve(
        args.midi,
        score_bpm=args.score_bpm,
        beat_unit=args.beat_unit,
        track_ids=[args.track] if args.track is not None else None,
        smoothing_window_beats=args.smoothing,
        polyorder=args.polyorder,
        use_librosa=args.use_librosa,
        as_json=args.as_json,
        return_df=False,
        out_path=dest,
    )
    if logger.isEnabledFor(logging.INFO):
        if args.as_json:
            tf = pd.read_json(out_path)["tempo_factor"]
        else:
            tf = pd.read_csv(out_path, usecols=["tempo_factor"])["tempo_factor"]
        logger.info(
            "Saved %s. tempo_factor stats -> min: %.4f max: %.4f mean: %.4f",
            out_path,
            tf.min(),
            tf.max(),
            tf.mean(),
        )


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    _cli()
