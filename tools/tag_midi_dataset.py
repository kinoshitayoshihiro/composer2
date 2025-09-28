# mypy: ignore-errors
# pyright: reportGeneralTypeIssues=false

"""Batch instrumentation and tagging analysis for MIDI files.

This utility inspects one or more MIDI files and produces lightweight
metadata describing:

* Instrument families present in each file (tracks and note distribution)
* Per-bar section and intensity estimates, mirroring ``data_ops.auto_tag``
* Mood/inensity heuristics compatible with ``tags.yaml`` entries
* Tempo / time-signature fallbacks for downstream DUV conditioning

The script is designed as a *prototype* for the upcoming data enrichment
pipeline.  It intentionally depends only on readily available statistics so
that it can run before any neural checkpoints are trained or downloaded.

Example usage::

    python -m tools.tag_midi_dataset data/midi --out metadata/tags.yaml

"""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, TypedDict, cast

import numpy as np
import yaml
from numpy.typing import NDArray

from utilities.duv_infer import (
    OPTIONAL_COLUMNS,
    OPTIONAL_FLOAT32_COLUMNS,
    OPTIONAL_INT32_COLUMNS,
    REQUIRED_COLUMNS,
)
from utilities.midi_utils import safe_end_time
from utilities.time_utils import seconds_to_qlen
from utilities import vocal_sync

try:  # pragma: no cover - optional dependency is part of project extras
    import pretty_midi  # type: ignore[import]
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise SystemExit("pretty_midi is required for tag_midi_dataset") from exc

PM_LIB = cast(Any, pretty_midi)

try:  # pragma: no cover - optional dependency
    from sklearn.cluster import KMeans
except ImportError:
    KMeans = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from hmmlearn.hmm import GaussianHMM  # type: ignore[import]
except ImportError:
    GaussianHMM = None  # type: ignore


LOGGER = logging.getLogger("tag_midi_dataset")
SECTION_NAMES = ("intro", "verse", "chorus", "bridge")
INTENSITY_LEVELS = ("low", "mid", "high")

DUV_GRID_DEFAULT = 0.25  # 1/16 note in beats (assuming 4/4)
VOCAL_ONSET_LIMIT = 128
VOCAL_REST_LIMIT = 64
DUV_OPTIONAL_REPORT = {
    "bar",
    "track_id",
    "program",
    "vel_bucket",
    "dur_bucket",
    "pitch_class",
    "start",
    "onset",
    "q_onset",
    "q_duration",
    "section",
    "mood",
}
DUV_OPTIONAL_UNIVERSE = OPTIONAL_COLUMNS | OPTIONAL_FLOAT32_COLUMNS | OPTIONAL_INT32_COLUMNS


def _safe_round(value: float, places: int = 6) -> float:
    rounded = round(float(value), places)
    # Avoid returning -0.0 when rounding small negatives.
    return 0.0 if abs(rounded) < 10 ** (-places) else rounded


def _parse_grid_arg(
    value: str | float | None,
    *,
    default: float = DUV_GRID_DEFAULT,
) -> float:
    """Parse CLI-friendly grids like ``1/16`` or ``0.25`` into beats."""

    if value is None:
        return default
    if isinstance(value, (int, float)):
        grid = float(value)
    else:
        text = str(value).strip()
        if not text:
            return default
        if "/" in text:
            num_str, denom_str = text.split("/", 1)
            try:
                num_val = float(num_str)
                denom_val = float(denom_str)
            except ValueError as exc:  # pragma: no cover - defensive parsing
                raise ValueError(f"invalid grid fraction: {value!r}") from exc
            if denom_val == 0:
                raise ValueError("grid denominator cannot be zero")
            # Convert whole-note fraction (e.g. 1/16) to beats (quarter notes).
            grid = (num_val / denom_val) * 4.0
        else:
            try:
                grid = float(text)
            except ValueError as exc:  # pragma: no cover - defensive parsing
                raise ValueError(f"invalid grid value: {value!r}") from exc
    if grid <= 0:
        raise ValueError("grid must be positive")
    return grid


def _bucket_velocity(velocity: float) -> int:
    return max(0, min(15, int(round(float(velocity) / 8.0))))


def _bucket_duration(duration_beats: float, grid_beats: float) -> int:
    if grid_beats <= 0:
        return 0
    steps = int(round(duration_beats / grid_beats))
    return max(0, min(63, steps))


def _collect_note_records(
    pm: Any,
    grid_beats: float,
    *,
    section_map: Mapping[int, str] | None = None,
    mood_label: str | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for track_idx, inst in enumerate(pm.instruments):
        if not inst.notes:
            continue
        program = int(inst.program) if not inst.is_drum else 0
        for note in inst.notes:
            start_sec = float(note.start)
            end_sec = float(note.end)
            start_qlen = float(seconds_to_qlen(pm, start_sec))
            end_qlen = float(seconds_to_qlen(pm, end_sec))
            duration_qlen = max(end_qlen - start_qlen, 0.0)
            bar_idx = int(math.floor(start_qlen / 4.0))
            beat_in_bar = start_qlen - (bar_idx * 4.0)
            if grid_beats > 0:
                position = int(round(beat_in_bar / grid_beats))
            else:
                position = int(round(beat_in_bar))
            vel_bucket = _bucket_velocity(note.velocity)
            dur_bucket = _bucket_duration(duration_qlen, grid_beats)
            record: dict[str, Any] = {
                "track_id": int(track_idx),
                "program": int(program),
                "pitch": int(note.pitch),
                "pitch_class": int(note.pitch % 12),
                "velocity": float(note.velocity),
                "duration": _safe_round(duration_qlen, 6),
                "bar": int(bar_idx),
                "position": int(position),
                "start": _safe_round(start_sec, 6),
                "onset": _safe_round(start_sec, 6),
                "q_onset": _safe_round(start_qlen, 6),
                "q_duration": _safe_round(duration_qlen, 6),
                "vel_bucket": int(vel_bucket),
                "dur_bucket": int(dur_bucket),
            }
            if section_map is not None:
                section_name = section_map.get(bar_idx)
                if section_name is not None:
                    record["section"] = str(section_name)
            if mood_label is not None:
                record["mood"] = str(mood_label)
            records.append(record)
    return records


def compute_duv_summary(
    pm: Any,
    *,
    grid_beats: float = DUV_GRID_DEFAULT,
    preview_limit: int = 12,
    section_map: Mapping[int, str] | None = None,
    mood_label: str | None = None,
) -> dict[str, Any]:
    note_records = _collect_note_records(
        pm,
        grid_beats,
        section_map=section_map,
        mood_label=mood_label,
    )
    summary: dict[str, Any] = {
        "duv_ready": bool(note_records),
        "grid_beats": _safe_round(grid_beats, 4),
        "note_count": int(len(note_records)),
        "required_columns": sorted(REQUIRED_COLUMNS),
    }
    if not note_records:
        summary["optional_present"] = []
        summary["optional_missing"] = sorted(DUV_OPTIONAL_REPORT)
        return summary

    columns: set[str] = set()
    velocities: list[float] = []
    durations: list[float] = []
    bars: list[int] = []
    pitch_classes: Counter[int] = Counter()
    vel_buckets: Counter[int] = Counter()
    dur_buckets: Counter[int] = Counter()
    for rec in note_records:
        columns.update(rec.keys())
        velocities.append(float(rec["velocity"]))
        durations.append(float(rec["q_duration"]))
        bars.append(int(rec["bar"]))
        pitch_classes.update([int(rec["pitch_class"])])
        vel_buckets.update([int(rec["vel_bucket"])])
        dur_buckets.update([int(rec["dur_bucket"])])

    unique_bars = sorted(set(bars))
    velocity_mean = _safe_round(float(np.mean(velocities)), 3)
    velocity_median = _safe_round(float(np.median(velocities)), 3)
    velocity_std = _safe_round(float(np.std(velocities)), 3)
    duration_mean = _safe_round(float(np.mean(durations)), 3)
    duration_median = _safe_round(float(np.median(durations)), 3)
    duration_std = _safe_round(float(np.std(durations)), 3)
    summary.update(
        {
            "bars_covered": int(len(unique_bars)),
            "max_bar_index": int(unique_bars[-1]) if unique_bars else -1,
            "velocity_mean": velocity_mean,
            "velocity_median": velocity_median,
            "velocity_std": velocity_std,
            "duration_mean_beats": duration_mean,
            "duration_median_beats": duration_median,
            "duration_std_beats": duration_std,
            "pitch_class_counts": {int(k): int(v) for k, v in pitch_classes.items()},
            "velocity_bucket_counts": {int(k): int(v) for k, v in vel_buckets.items()},
            "duration_bucket_counts": {int(k): int(v) for k, v in dur_buckets.items()},
        }
    )

    optional_present = sorted(col for col in columns if col in DUV_OPTIONAL_REPORT)
    optional_missing = sorted(col for col in DUV_OPTIONAL_REPORT if col not in columns)
    unused_optionals = sorted(
        col
        for col in DUV_OPTIONAL_UNIVERSE - set(optional_present)
        if col in {"section", "mood", "vel_bucket", "dur_bucket"}
    )
    summary["optional_present"] = optional_present
    if optional_missing:
        summary["optional_missing"] = optional_missing
    if unused_optionals:
        summary["optional_available_not_populated"] = unused_optionals

    preview = sorted(
        note_records,
        key=lambda rec: (rec["bar"], rec["position"], rec["pitch"]),
    )[: max(0, preview_limit)]
    summary["preview"] = preview
    return summary


def compute_vocal_sync_summary(
    pm: Any,
    *,
    tempo_bpm: float,
    track_index: int,
    grid_beats: float,
) -> dict[str, Any] | None:
    if grid_beats <= 0:
        grid_beats = DUV_GRID_DEFAULT
    sync_module = cast(Any, vocal_sync)
    try:
        onsets = cast(
            list[float],
            sync_module.extract_onsets(
                pm,
                tempo=tempo_bpm,
                track_idx=track_index,
            ),
        )
    except (RuntimeError, ValueError, AttributeError) as exc:
        LOGGER.debug(
            "Vocal timing extraction failed for track %d: %s",
            track_index,
            exc,
        )
        return None

    if not onsets:
        return None

    quantized = sync_module.quantize_times(onsets, grid=grid_beats, dedup=True)
    quantized = [_safe_round(val, 4) for val in quantized]
    rests_raw = sync_module.extract_long_rests(onsets, min_rest=grid_beats)
    rest_entries: list[dict[str, float]] = []
    for start, duration in rests_raw:
        q_start = sync_module.quantize_times([start], grid=grid_beats)[0]
        rest_entries.append(
            {
                "start": _safe_round(q_start, 4),
                "duration": _safe_round(duration, 4),
            }
        )

    return {
        "track_index": int(track_index),
        "tempo_bpm": _safe_round(tempo_bpm, 3),
        "grid_beats": _safe_round(grid_beats, 4),
        "onset_count": int(len(onsets)),
        "raw_preview": [_safe_round(val, 4) for val in onsets[:16]],
        "quantized_onsets": quantized[:VOCAL_ONSET_LIMIT],
        "rests": rest_entries[:VOCAL_REST_LIMIT],
    }


@dataclass(slots=True)
class MidiFeatures:
    path: Path
    pm: Any
    bar_density: NDArray[np.float32]
    bar_velocity: NDArray[np.float32]
    bar_energy: NDArray[np.float32]
    avg_velocity: float
    total_notes: int
    duration_seconds: float


class FamilyStats(TypedDict):
    tracks: int
    notes: int


class ProgramStats(TypedDict):
    tracks: int
    notes: int


class FamilySummary(TypedDict):
    name: str
    tracks: int
    note_ratio: float


class ProgramSummary(TypedDict):
    name: str
    tracks: int
    notes: int


class InstrumentSummary(TypedDict):
    families: list[FamilySummary]
    programs: list[ProgramSummary]
    primary_family: str | None
    primary_program: str | None


def collect_midi_files(target: Path, recursive: bool) -> list[Path]:
    """Return sorted list of MIDI file paths under ``target``."""

    if target.is_file():
        if target.suffix.lower() not in {".mid", ".midi"}:
            raise ValueError(f"Unsupported file extension: {target}")
        return [target]

    patterns = ("*.mid", "*.midi")
    globber = Path.rglob if recursive else Path.glob
    files: list[Path] = []
    for pattern in patterns:
        files.extend(sorted(globber(target, pattern)))
    unique = sorted(dict.fromkeys(files))
    LOGGER.info("Found %d MIDI files", len(unique))
    return unique


def extract_bar_features(
    pm: Any,
) -> tuple[
    NDArray[np.int32],
    NDArray[np.int32],
    NDArray[np.int32],
]:
    """Compute per-bar density, median velocity, and energy."""

    _, tempos = pm.get_tempo_changes()
    tempo = float(tempos[0]) if tempos.size else 120.0
    tempo_fallback = False
    if tempo <= 0:
        # ``estimate_tempo`` returns a scalar and handles silent files
        # gracefully.
        tempo = float(pm.estimate_tempo() or 120.0)
        tempo_fallback = True
    beat = 60.0 / tempo
    # Determine bars via a basic 4/4 assumption; downstream consumers may
    # override as needed.
    bar_length = beat * 4
    end_time = safe_end_time(pm)
    num_bars = max(1, int(round(end_time / bar_length)))

    density: list[int] = []
    velocity: list[int] = []
    energy: list[int] = []
    for idx in range(num_bars):
        start = idx * bar_length
        end = start + bar_length
        notes = [
            note for inst in pm.instruments for note in inst.notes if start <= note.start < end
        ]
        density.append(len(notes))
        if notes:
            vel_vals = [int(note.velocity) for note in notes]
            velocity.append(int(np.median(vel_vals)))
            energy.append(int(sum(vel_vals)))
        else:
            velocity.append(0)
            energy.append(0)

    feats = (
        np.array(density, dtype=np.int32),
        np.array(velocity, dtype=np.int32),
        np.array(energy, dtype=np.int32),
    )
    if tempo_fallback:
        LOGGER.debug("Tempo fallback invoked for silent or zero-tempo file")
    return feats


def summarise_instruments(pm: Any) -> InstrumentSummary:
    """Summarise instrument families and top programs for ``pm``."""

    families: dict[str, FamilyStats] = {}
    programs: dict[str, ProgramStats] = {}
    total_notes = 0
    for inst in pm.instruments:
        note_count = len(inst.notes)
        if note_count == 0:
            continue
        total_notes += note_count
        if inst.is_drum:
            family_name = "drums"
            program_name = "Drum Kit"
        else:
            program_raw = PM_LIB.program_to_instrument_name(int(inst.program))
            program_name = str(program_raw)
            family_raw = PM_LIB.program_to_instrument_class(int(inst.program))
            family_name = str(family_raw).lower().replace(" ", "_")

        if family_name not in families:
            families[family_name] = {"tracks": 0, "notes": 0}
        if program_name not in programs:
            programs[program_name] = {"tracks": 0, "notes": 0}

        families[family_name]["tracks"] += 1
        families[family_name]["notes"] += note_count
        programs[program_name]["tracks"] += 1
        programs[program_name]["notes"] += note_count

    if total_notes == 0:
        return InstrumentSummary(
            families=[],
            programs=[],
            primary_family=None,
            primary_program=None,
        )

    family_summary: list[FamilySummary] = []
    for name, stats in sorted(
        families.items(),
        key=lambda item: item[1]["notes"],
        reverse=True,
    ):
        family_summary.append(
            {
                "name": name,
                "tracks": stats["tracks"],
                "note_ratio": round(stats["notes"] / total_notes, 4),
            }
        )

    program_summary: list[ProgramSummary] = []
    for name, stats in sorted(
        programs.items(),
        key=lambda item: item[1]["notes"],
        reverse=True,
    ):
        program_summary.append(
            {
                "name": name,
                "tracks": stats["tracks"],
                "notes": stats["notes"],
            }
        )

    primary_family: str | None = family_summary[0]["name"] if family_summary else None
    primary_program: str | None = program_summary[0]["name"] if program_summary else None

    return InstrumentSummary(
        families=family_summary,
        programs=program_summary,
        primary_family=primary_family,
        primary_program=primary_program,
    )


def load_features(files: Sequence[Path]) -> list[MidiFeatures]:
    """Load MIDI files and compute aggregate statistics."""

    analyses: list[MidiFeatures] = []
    for path in files:
        try:
            pm = cast(Any, pretty_midi.PrettyMIDI(str(path)))
        except (OSError, ValueError) as exc:  # pragma: no cover
            LOGGER.warning("Failed to load %s: %s", path, exc)
            continue

        dens_raw, vels_raw, eng_raw = extract_bar_features(pm)
        dens = dens_raw.astype(np.float32)
        vels = vels_raw.astype(np.float32)
        eng = eng_raw.astype(np.float32)
        velocities = [note.velocity for inst in pm.instruments for note in inst.notes]
        avg_velocity = float(np.mean(velocities)) if velocities else 0.0
        total_notes = sum(len(inst.notes) for inst in pm.instruments)
        duration = safe_end_time(pm)
        analyses.append(
            MidiFeatures(
                path=path,
                pm=pm,
                bar_density=dens,
                bar_velocity=vels,
                bar_energy=eng,
                avg_velocity=avg_velocity,
                total_notes=total_notes,
                duration_seconds=duration,
            )
        )

    return analyses


def cluster_intensity(
    all_features: NDArray[np.float32],
    n_clusters: int,
) -> tuple[NDArray[np.int32], dict[int, str]]:
    """Cluster per-bar features into intensity levels."""

    if all_features.size == 0:
        return np.zeros(0, dtype=np.int32), {}

    if KMeans is None or all_features.shape[0] < n_clusters:
        LOGGER.warning(
            "KMeans unavailable or feature count < clusters; " "falling back to velocity quantiles",
        )
        velocities = all_features[:, 1].astype(np.float32, copy=False)
        if velocities.size == 0:
            return np.zeros(0, dtype=np.int32), {}
        if velocities.size > 2:
            quantiles = np.quantile(velocities, [0.33, 0.66])
            quantiles = quantiles.astype(np.float32, copy=False)
            q_low = float(quantiles[0])
            q_mid = float(quantiles[1])
        else:
            median = float(np.median(velocities))
            q_low = median
            q_mid = median
        labels = np.zeros(int(velocities.size), dtype=np.int32)
        for idx, vel in enumerate(velocities.tolist()):
            if vel <= q_low:
                labels[idx] = 0
            elif vel <= q_mid:
                labels[idx] = 1
            else:
                labels[idx] = 2
        return labels.astype(np.int32), {0: "low", 1: "mid", 2: "high"}

    kmeans = cast(Any, KMeans(n_clusters=n_clusters, random_state=0))
    numeric = np.asarray(kmeans.fit_predict(all_features), dtype=np.int32)
    centers = np.asarray(kmeans.cluster_centers_, dtype=np.float32)
    ordering = sorted(
        ((float(center.sum()), idx) for idx, center in enumerate(centers)),
        key=lambda item: item[0],
    )
    label_map: dict[int, str] = {}
    for pos, (_, idx) in enumerate(ordering):
        level = INTENSITY_LEVELS[min(pos, len(INTENSITY_LEVELS) - 1)]
        label_map[idx] = level
    return numeric, label_map


def assign_sections(all_densities: NDArray[np.float32]) -> NDArray[np.int32]:
    """Predict section indices for concatenated bar densities."""

    if all_densities.size == 0:
        return np.zeros(0, dtype=np.int32)

    if GaussianHMM is not None and all_densities.size >= 4:
        try:
            hmm = cast(
                Any,
                GaussianHMM(n_components=4, random_state=0, n_iter=16),
            )
            reshaped = all_densities.reshape(-1, 1)
            hmm.fit(reshaped)
            predicted = hmm.predict(reshaped)
            return np.asarray(predicted, dtype=np.int32)
        except (ValueError, RuntimeError) as exc:  # pragma: no cover
            LOGGER.warning(
                "GaussianHMM failed (%s); using modulo fallback",
                exc,
            )

    idx = np.arange(all_densities.size, dtype=np.int32)
    return np.mod(idx, 4, dtype=np.int32)


def infer_mood(
    intensities: Sequence[str],
    avg_velocity: float,
) -> tuple[str, float]:
    """Infer mood label and confidence score from intensity sequence."""

    if not intensities:
        return "neutral", 0.0

    counts = Counter(intensities)
    dominant, count = counts.most_common(1)[0]
    confidence = count / sum(counts.values()) if counts else 0.0

    base_map = {
        "low": "calm",
        "mid": "melancholic",
        "high": "energetic",
    }
    mood = base_map.get(dominant, "neutral")

    if avg_velocity < 45:
        mood = "calm"
    elif avg_velocity > 95:
        mood = "energetic"
    elif mood == "calm" and avg_velocity > 60:
        mood = "melancholic"

    return mood, round(confidence, 3)


def derive_mode(instr_summary: InstrumentSummary) -> str:
    """Return overall mode descriptor based on instrument distribution."""

    families = instr_summary["families"]
    if not families:
        return "unknown"

    drums_only = all(item["name"] == "drums" for item in families)
    if drums_only:
        return "drums"

    if families[0]["name"] == "strings" and len(families) == 1:
        return "solo_strings"

    return "ensemble"


def analyse_dataset(
    features: list[MidiFeatures],
    *,
    intensity_clusters: int,
    base_path: Path | None,
    duv_grid: float,
    duv_preview: int,
    vocal_track: int | None,
    vocal_grid: float,
) -> dict[str, dict[str, object]]:
    """Combine file features into metadata for export."""

    if not features:
        return {}

    feature_blocks = [
        np.stack(
            (
                feat.bar_density.astype(np.float32),
                feat.bar_velocity.astype(np.float32),
                feat.bar_energy.astype(np.float32),
            ),
            axis=1,
        )
        for feat in features
    ]
    concatenated = np.concatenate(feature_blocks, axis=0)
    density_stack = np.concatenate(
        [feat.bar_density for feat in features],
        axis=0,
    )

    intensity_numeric, label_map = cluster_intensity(
        concatenated,
        intensity_clusters,
    )
    section_indices = assign_sections(density_stack)

    results: dict[str, dict[str, object]] = {}
    offset = 0
    section_offset = 0
    for feat in features:
        bars = len(feat.bar_density)
        if bars == 0:
            LOGGER.debug("Skipping %s due to zero bar count", feat.path)
            continue

        idx_slice = slice(offset, offset + bars)
        sec_slice = slice(section_offset, section_offset + bars)
        offset += bars
        section_offset += bars

        numeric_labels = intensity_numeric[idx_slice]
        intensity_seq = [
            label_map.get(
                int(label),
                INTENSITY_LEVELS[min(int(label), len(INTENSITY_LEVELS) - 1)],
            )
            for label in numeric_labels
        ]
        section_seq = [
            SECTION_NAMES[int(idx) % len(SECTION_NAMES)] for idx in section_indices[sec_slice]
        ]

        intensity_counts = Counter(intensity_seq)
        dominant_intensity, intensity_freq = intensity_counts.most_common(1)[0]
        intensity_conf = round(intensity_freq / bars, 3)

        section_counts = Counter(section_seq)
        dominant_section, section_freq = section_counts.most_common(1)[0]
        section_conf = round(section_freq / bars, 3)

        mood, mood_conf = infer_mood(intensity_seq, feat.avg_velocity)
        instr_summary = summarise_instruments(feat.pm)
        mode = derive_mode(instr_summary)

        tempo_changes = feat.pm.get_tempo_changes()
        tempos = tempo_changes[1]
        tempo_fallback = bool(tempos.size == 0 or tempos[0] <= 0)
        bpm = float(tempos[0]) if tempos.size else float(feat.pm.estimate_tempo() or 120.0)

        # PrettyMIDI's time signature changes lack stable typing across
        # versions; default to 4/4 and let downstream tooling override.
        ts_string = "4/4"
        ts_fallback = True

        if base_path and feat.path.is_relative_to(base_path):
            key = str(feat.path.relative_to(base_path))
        else:
            key = feat.path.name

        metadata: dict[str, object] = {
            "bars": float(bars),
            "bpm": round(bpm, 3),
            "tempo_fallback": tempo_fallback,
            "time_signature": ts_string,
            "ts_fallback": ts_fallback,
            "intensity": dominant_intensity,
            "confidence_intensity": intensity_conf,
            "section": dominant_section,
            "confidence_section": section_conf,
            "mood": mood,
            "confidence_mood": mood_conf,
            "mode": mode,
            "avg_velocity": round(feat.avg_velocity, 3),
            "total_notes": int(feat.total_notes),
            "duration_seconds": round(feat.duration_seconds, 3),
            "sections": {
                "sequence": section_seq,
                "counts": dict(section_counts),
            },
            "intensity_sequence": intensity_seq,
            "instrumentation": instr_summary,
        }

        section_map = dict(enumerate(section_seq))

        duv_summary = compute_duv_summary(
            feat.pm,
            grid_beats=duv_grid,
            preview_limit=duv_preview,
            section_map=section_map,
            mood_label=mood,
        )
        metadata["duv_summary"] = duv_summary

        if vocal_track is not None:
            vocal_data = compute_vocal_sync_summary(
                feat.pm,
                tempo_bpm=bpm,
                track_index=vocal_track,
                grid_beats=vocal_grid,
            )
            if vocal_data:
                metadata["vocal_sync"] = vocal_data

        results[key] = metadata

    return results


def normalise_for_serialisation(data: Any) -> Any:
    """Convert numpy scalar/array types into JSON/YAML friendly values."""

    if isinstance(data, np.generic):
        return data.item()
    if isinstance(data, np.ndarray):
        return [normalise_for_serialisation(item) for item in data.tolist()]
    if isinstance(data, Mapping):
        result: dict[str, Any] = {}
        for key, value in cast(Mapping[Any, Any], data).items():
            result[str(key)] = normalise_for_serialisation(value)
        return result
    if isinstance(data, (list, tuple)):
        return [normalise_for_serialisation(item) for item in cast(Iterable[Any], data)]
    if isinstance(data, set):
        return [normalise_for_serialisation(item) for item in cast(Iterable[Any], data)]
    return data


def write_output(
    data: dict[str, dict[str, object]],
    out_path: Path,
    fmt: str,
    *,
    pretty: bool,
) -> None:
    """Serialize ``data`` to YAML or JSON."""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = normalise_for_serialisation(data)
    if fmt == "json":
        indent = 2 if pretty else None
        out_path.write_text(
            json.dumps(payload, indent=indent, ensure_ascii=False),
            encoding="utf-8",
        )
    else:
        # Default YAML keeps order for readability.
        out_path.write_text(
            yaml.safe_dump(
                payload,
                sort_keys=False,
                allow_unicode=True,
                indent=2,
            ),
            encoding="utf-8",
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch instrumentation & tagging for MIDI datasets",
    )
    parser.add_argument(
        "path",
        type=Path,
        help="MIDI file or directory to analyse",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("tags.generated.yaml"),
        help="Output file path",
    )
    parser.add_argument(
        "--format",
        choices=("yaml", "json"),
        default="yaml",
        help="Output format",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan directories for MIDI files",
    )
    parser.add_argument(
        "--k-intensity",
        type=int,
        default=3,
        help="Number of clusters for intensity analysis",
    )
    parser.add_argument(
        "--relative-to",
        type=Path,
        default=None,
        help="Base path for output keys",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output",
    )
    parser.add_argument(
        "--duv-grid",
        default="1/16",
        help="Quantisation grid (beats or fraction) for DUV summaries",
    )
    parser.add_argument(
        "--duv-preview",
        type=int,
        default=12,
        help="Number of note rows to include in the DUV preview",
    )
    parser.add_argument(
        "--vocal-track",
        type=int,
        help=(
            "Optional track index analysed via VocalSynchro utilities; "
            "omit to skip vocal timing output"
        ),
    )
    parser.add_argument(
        "--vocal-grid",
        default="1/16",
        help="Quantisation grid for vocal timing summaries",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (e.g. INFO, DEBUG)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(levelname)s %(message)s",
    )

    files = collect_midi_files(args.path, recursive=args.recursive)
    if not files:
        raise SystemExit("No MIDI files discovered")

    try:
        duv_grid = _parse_grid_arg(args.duv_grid)
    except ValueError as exc:
        raise SystemExit(f"--duv-grid: {exc}") from exc
    try:
        vocal_grid = _parse_grid_arg(args.vocal_grid)
    except ValueError as exc:
        raise SystemExit(f"--vocal-grid: {exc}") from exc

    duv_preview = max(0, int(args.duv_preview))
    vocal_track = args.vocal_track

    features = load_features(files)
    metadata = analyse_dataset(
        features,
        intensity_clusters=args.k_intensity,
        base_path=args.relative_to,
        duv_grid=duv_grid,
        duv_preview=duv_preview,
        vocal_track=vocal_track,
        vocal_grid=vocal_grid,
    )
    if not metadata:
        raise SystemExit("No analyzable MIDI files found")

    write_output(metadata, args.out, args.format, pretty=args.pretty)
    LOGGER.info("Wrote %d entries to %s", len(metadata), args.out)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
