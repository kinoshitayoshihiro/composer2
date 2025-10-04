#!/usr/bin/env python3
"""Extract bass MIDI stems from a Slakh2100 MIDI dump.

The Slakh2100 Redux archive stores each song inside its own directory with
per-stem MIDI files and a ``metadata.yaml`` descriptor.  This utility scans the
extracted archive, identifies stems whose General MIDI program indicates an
electric or acoustic bass (program numbers 32-39 by default), and re-exports
them into a consolidated directory for downstream training.

A CSV manifest describing every extracted file is emitted alongside the MIDI
clips so downstream pipelines (for example our DUV data preparation scripts)
can reuse pitch/velocity statistics without re-scanning the full dataset.
"""

from __future__ import annotations

import argparse
import copy
import csv
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, cast

import pretty_midi as pm  # type: ignore[import-untyped]
import yaml

# General MIDI programs corresponding to bass instruments.
DEFAULT_GM_BASS_PROGRAMS = tuple(range(32, 40))
DEFAULT_MIN_NOTES = 8

InstrumentLike = Any

LOGGER = logging.getLogger("extract_slakh_bass")


@dataclass(slots=True)
class ExtractionResult:
    """Information about one extracted bass stem."""

    song_id: str
    stem_id: str
    instrument_name: str
    midi_source: Path
    midi_output: Path
    program: Optional[int]
    note_count: int
    total_duration: float


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan an extracted Slakh2100 directory and collect bass stems "
            "into a dedicated output folder."
        ),
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/slakh2100_midi"),
        help=(
            "Root directory produced by scripts/download_slakh_midi.py. "
            "This path should contain per-song folders with metadata.yaml "
            "files."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/slakh2100_bass"),
        help=("Destination directory where extracted bass-only MIDI files are " "stored."),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/slakh2100_bass_manifest.csv"),
        help=("CSV file summarising all extracted stems (default: %(default)s)."),
    )
    parser.add_argument(
        "--gm-programs",
        type=int,
        nargs="*",
        default=list(DEFAULT_GM_BASS_PROGRAMS),
        help=(
            "General MIDI program numbers that should be treated as bass. "
            "Defaults to 32-39 inclusive."
        ),
    )
    parser.add_argument(
        "--min-notes",
        type=int,
        default=DEFAULT_MIN_NOTES,
        help=(
            "Minimum number of notes required to keep a stem. Tracks with "
            "fewer notes are skipped (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing extracted MIDI files if they already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=("Scan and report what would be extracted without writing any " "files."),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging for troubleshooting.",
    )
    return parser.parse_args(argv)


def normalise_song_id(metadata_path: Path, payload: dict[str, object]) -> str:
    song_id = payload.get("track_id") or payload.get("song_id") or payload.get("mix_id")
    if isinstance(song_id, str) and song_id.strip():
        return song_id.strip()
    return metadata_path.parent.name


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "stem"


def is_bass_program(
    program: Optional[int],
    gm_programs: Iterable[int],
) -> bool:
    return program is not None and int(program) in gm_programs


def is_bass_name(name: Optional[str]) -> bool:
    if not name:
        return False
    lowered = name.lower()
    keywords = ("bass", "upright", "contrabass")
    return any(keyword in lowered for keyword in keywords)


def resolve_midi_path(
    stem_meta: dict[str, object],
    metadata_path: Path,
    stem_id: Optional[str] = None,
) -> Optional[Path]:
    candidates = [
        stem_meta.get("midi_filename"),
        stem_meta.get("midi_file"),
        stem_meta.get("midi"),
        stem_meta.get("midi_path"),
    ]
    for candidate in candidates:
        if isinstance(candidate, str) and candidate:
            path = metadata_path.parent / candidate
            if path.exists():
                return path
    if stem_id:
        midi_dir = metadata_path.parent / "MIDI"
        fallback = midi_dir / f"{stem_id}.mid"
        if fallback.exists():
            return fallback
    return None


def parse_program(value: object) -> Optional[int]:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def pick_bass_instruments(
    midi: pm.PrettyMIDI,
    gm_programs: Iterable[int],
) -> list[InstrumentLike]:
    midi_any = cast(Any, midi)
    instruments = cast(
        list[InstrumentLike],
        getattr(midi_any, "instruments", []),
    )
    bass_instr: list[InstrumentLike] = []
    for inst_any in instruments:
        if getattr(inst_any, "is_drum", False):
            continue
        program_val = getattr(inst_any, "program", None)
        program = parse_program(program_val)
        name_val = getattr(inst_any, "name", None)
        if is_bass_program(program, gm_programs) or is_bass_name(str(name_val)):
            bass_instr.append(inst_any)
    return bass_instr


def extract_bass_from_metadata(
    metadata_path: Path,
    gm_programs: Iterable[int],
    min_notes: int,
    dry_run: bool,
    overwrite: bool,
    output_root: Path,
) -> list[ExtractionResult]:
    with metadata_path.open("r", encoding="utf-8") as handle:
        raw_payload: Any = yaml.safe_load(handle) or {}

    if not isinstance(raw_payload, dict):
        LOGGER.warning("Unexpected metadata format at %s", metadata_path)
        return []

    payload = cast(dict[str, object], raw_payload)

    stems_raw = payload.get("stems")
    if not isinstance(stems_raw, dict):
        LOGGER.debug("No stems entry in metadata %s", metadata_path)
        return []
    stems = cast(dict[str, object], stems_raw)

    song_id = normalise_song_id(metadata_path, payload)
    results: list[ExtractionResult] = []

    for stem_id, stem_meta_raw in stems.items():
        if not isinstance(stem_meta_raw, dict):
            continue
        stem_meta = cast(dict[str, object], stem_meta_raw)

        program_val = stem_meta.get("program_num") or stem_meta.get("program")
        program = parse_program(program_val)
        instrument_name_val = stem_meta.get("name") or stem_meta.get("instrument") or ""
        instrument_name = str(instrument_name_val)

        if not (is_bass_program(program, gm_programs) or is_bass_name(instrument_name)):
            continue

        midi_path = resolve_midi_path(
            stem_meta,
            metadata_path,
            stem_id=str(stem_id),
        )
        if midi_path is None:
            LOGGER.warning(
                "MIDI file missing for %s/%s",
                metadata_path.parent,
                stem_id,
            )
            continue

        try:
            midi = pm.PrettyMIDI(str(midi_path))
        except (OSError, ValueError) as exc:  # pragma: no cover
            # Corrupted MIDI or decode issue.
            LOGGER.warning("Failed to parse %s: %s", midi_path, exc)
            continue

        bass_instruments_raw = pick_bass_instruments(midi, gm_programs)
        bass_instruments_list: list[InstrumentLike] = list(bass_instruments_raw)
        if not bass_instruments_list:
            # Fallback: keep the entire file if it's single-instrument and the
            # metadata still marks it as bass.
            instruments_attr = cast(
                list[InstrumentLike],
                cast(Any, midi).instruments,
            )
            if len(instruments_attr) == 1:
                bass_instruments_list = instruments_attr
            else:
                LOGGER.debug("No bass instruments found in %s", midi_path)
                continue

        note_count = 0
        total_duration = 0.0
        for inst in bass_instruments_list:
            notes = getattr(inst, "notes", [])
            note_count += len(notes)
            for note in notes:
                end = float(getattr(note, "end", 0.0))
                start = float(getattr(note, "start", 0.0))
                total_duration += max(0.0, end - start)

        if note_count < min_notes:
            LOGGER.debug(
                "Skipping %s (only %d notes < min-notes %d)",
                midi_path,
                note_count,
                min_notes,
            )
            continue

        bass_midi = copy.deepcopy(midi)
        bass_midi_cast = cast(Any, bass_midi)
        bass_midi_cast.instruments = [copy.deepcopy(inst) for inst in bass_instruments_list]

        safe_name = slugify(instrument_name or str(stem_id))
        output_path = output_root / song_id / f"{stem_id}_{safe_name}.mid"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not dry_run:
            if output_path.exists() and not overwrite:
                LOGGER.debug("Skipping existing %s", output_path)
            else:
                cast(Any, bass_midi).write(str(output_path))

        results.append(
            ExtractionResult(
                song_id=song_id,
                stem_id=str(stem_id),
                instrument_name=instrument_name,
                midi_source=midi_path,
                midi_output=output_path,
                program=program,
                note_count=note_count,
                total_duration=total_duration,
            )
        )

    return results


def write_manifest(manifest_path: Path, rows: list[ExtractionResult]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "song_id",
                "stem_id",
                "instrument_name",
                "program",
                "note_count",
                "total_duration",
                "midi_source",
                "midi_output",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.song_id,
                    row.stem_id,
                    row.instrument_name,
                    row.program if row.program is not None else "",
                    row.note_count,
                    f"{row.total_duration:.6f}",
                    row.midi_source.as_posix(),
                    row.midi_output.as_posix(),
                ]
            )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    if not args.source.exists():
        LOGGER.error("Source directory %s does not exist", args.source)
        return 1

    gm_programs = set(args.gm_programs)
    metadata_files = sorted(args.source.rglob("metadata.yaml"))
    if not metadata_files:
        LOGGER.error("No metadata.yaml files found under %s", args.source)
        return 1

    LOGGER.info(
        "Scanning %d metadata files for bass stems",
        len(metadata_files),
    )
    all_results: list[ExtractionResult] = []

    for metadata_path in metadata_files:
        results = extract_bass_from_metadata(
            metadata_path=metadata_path,
            gm_programs=gm_programs,
            min_notes=args.min_notes,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
            output_root=args.output,
        )
        all_results.extend(results)

    LOGGER.info("Collected %d bass stems", len(all_results))

    if not args.dry_run and all_results:
        write_manifest(args.manifest, all_results)
        LOGGER.info("Manifest written to %s", args.manifest)

    if not all_results:
        LOGGER.warning("No stems matched the bass criteria.")

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
