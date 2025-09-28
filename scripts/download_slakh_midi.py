#!/usr/bin/env python3
"""Fetch Slakh2100 while keeping only MIDI and metadata files."""
from __future__ import annotations

import argparse
import sys
import tarfile
from pathlib import Path
from typing import Optional, Sequence, IO

try:
    import requests
except ImportError:  # pragma: no cover - handled at runtime
    print(
        "The 'requests' package is required to run this script.",
        file=sys.stderr,
    )
    raise

BUFFER_SIZE = 1024 * 1024  # 1 MiB


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the Slakh2100 archive from Zenodo (or another URL) "
            "and extract only the MIDI and per-track metadata files."
        ),
    )
    parser.add_argument(
        "--url",
        default=(
            "https://zenodo.org/api/records/4599666/files/" "slakh2100_flac_redux.tar.gz/content"
        ),
        help=(
            "HTTP(S) URL of the tar.gz archive to stream. Defaults to the "
            "official Zenodo link for Slakh2100 Redux."
        ),
    )
    parser.add_argument(
        "--archive",
        type=Path,
        help="Optional local .tar.gz archive to read instead of downloading.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/slakh2100_midi"),
        help="Directory where extracted MIDI/metadata files will be stored.",
    )
    parser.add_argument(
        "--keep-metadata",
        action="store_true",
        help=("Also extract each track's metadata.yaml alongside the MIDI files."),
    )
    parser.add_argument(
        "--strip-components",
        type=int,
        default=1,
        help=("Number of leading path components to strip from archive entries " "(default: 1)."),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=("Skip files that already exist at the destination (based on the " "file path)."),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help=("Suppress per-file logging. Only summary information will be " "printed."),
    )
    return parser.parse_args(argv)


def should_extract(member: tarfile.TarInfo, include_metadata: bool) -> bool:
    name = member.name.lower()
    if member.isdir():
        return False
    if name.endswith(".mid"):
        return True
    if include_metadata and name.endswith("metadata.yaml"):
        return True
    return False


def normalized_target(member: tarfile.TarInfo, strip_components: int) -> Path:
    parts = Path(member.name).parts
    if strip_components >= len(parts):
        raise ValueError(
            (f"Cannot strip {strip_components} components from path " f"'{member.name}'")
        )
    return Path(*parts[strip_components:])


def extract_members(
    archive: tarfile.TarFile,
    output_dir: Path,
    include_metadata: bool,
    strip_components: int,
    resume: bool,
    quiet: bool,
) -> tuple[int, int]:
    extracted = 0
    skipped = 0
    for member in archive:
        if not should_extract(member, include_metadata):
            continue
        try:
            relative_path = normalized_target(member, strip_components)
        except ValueError:
            skipped += 1
            continue
        destination = output_dir / relative_path
        if resume and destination.exists():
            skipped += 1
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        file_obj = archive.extractfile(member)
        if file_obj is None:
            skipped += 1
            continue
        with file_obj, open(destination, "wb") as handle:
            copy_stream(file_obj, handle)
        extracted += 1
        if not quiet:
            print(f"[saved] {destination.relative_to(output_dir)}")
    return extracted, skipped


def copy_stream(src: IO[bytes], dst: IO[bytes]) -> None:
    while True:
        chunk = src.read(BUFFER_SIZE)
        if not chunk:
            break
        dst.write(chunk)


def open_archive_from_url(url: str) -> requests.Response:
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    response.raw.decode_content = True
    return response


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.archive:
        archive_path = args.archive.expanduser().resolve()
        if not archive_path.exists():
            print(f"Archive not found: {archive_path}", file=sys.stderr)
            return 1
        stream = archive_path.open("rb")
        response = None
    else:
        response = open_archive_from_url(args.url)
        stream = response.raw

    mode = "r|gz"
    stats = {}
    try:
        with tarfile.open(fileobj=stream, mode=mode) as archive:
            extracted, skipped = extract_members(
                archive,
                output_dir=output_dir,
                include_metadata=args.keep_metadata,
                strip_components=args.strip_components,
                resume=args.resume,
                quiet=args.quiet,
            )
            stats = {"extracted": extracted, "skipped": skipped}
    finally:
        stream.close()
        if response is not None:
            response.close()

    print(
        ("Extraction complete. Extracted {extracted} files, " "skipped {skipped}.").format(**stats)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
