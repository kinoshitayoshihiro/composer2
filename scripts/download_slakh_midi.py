#!/usr/bin/env python3
"""Fetch Slakh2100 while keeping only MIDI and metadata files."""
from __future__ import annotations

import argparse
import math
import sys
import tarfile
import time
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
DEFAULT_RETRY_WAIT = 5.0
DEFAULT_MAX_RETRIES = 5


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the Slakh2100 archive from Zenodo (or another URL) "
            "and extract only the MIDI and per-track metadata files."
        ),
    )
    parser.add_argument(
        "--url",
        default="https://zenodo.org/api/records/4599666/files/"
        "slakh2100_flac_redux.tar.gz/content",
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
        "--download-to",
        type=Path,
        help=(
            "Local file path to stage the remote archive before extraction. "
            "If the file already exists, the download will resume from the "
            "current size."
        ),
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
        "--max-download-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help=("Maximum number of retry attempts for staging downloads " "(default: %(default)s)."),
    )
    parser.add_argument(
        "--retry-wait",
        type=float,
        default=DEFAULT_RETRY_WAIT,
        help=("Seconds to wait between download retries (default: %(default)s)."),
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
        message = ("Cannot strip {count} components from path '{name}'").format(
            count=strip_components, name=member.name
        )
        raise ValueError(message)
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


def _format_size(num_bytes: int) -> str:
    if num_bytes <= 0:
        return "0 B"
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    index = min(int(math.log(num_bytes, 1024)), len(units) - 1)
    value = num_bytes / (1024**index)
    if value >= 100:
        formatted = f"{value:,.0f}"
    elif value >= 10:
        formatted = f"{value:,.1f}"
    else:
        formatted = f"{value:,.2f}"
    return f"{formatted} {units[index]}"


def download_archive(
    url: str,
    destination: Path,
    *,
    chunk_size: int = BUFFER_SIZE,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_wait: float = DEFAULT_RETRY_WAIT,
) -> Path:
    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()

    attempt = 0
    while True:
        attempt += 1
        existing = destination.stat().st_size if destination.exists() else 0
        headers: dict[str, str] = {}
        if existing:
            headers["Range"] = f"bytes={existing}-"
        try:
            with session.get(
                url,
                stream=True,
                timeout=60,
                headers=headers,
            ) as response:
                if response.status_code == 416:
                    return destination
                response.raise_for_status()
                total_size = response.headers.get("Content-Length")
                if total_size is not None:
                    total_size = int(total_size) + existing
                content_range = response.headers.get("Content-Range")
                if content_range and "/" in content_range:
                    try:
                        total_size = int(content_range.rsplit("/", 1)[-1])
                    except ValueError:
                        total_size = total_size

                mode = "ab" if existing else "wb"
                downloaded = existing
                start = time.monotonic()
                with open(destination, mode) as handle:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if not chunk:
                            continue
                        handle.write(chunk)
                        downloaded += len(chunk)
                        if total_size and sys.stderr.isatty():
                            elapsed = max(time.monotonic() - start, 1e-6)
                            speed = downloaded / elapsed
                            percent = downloaded / total_size * 100
                            message = (
                                "\r[download] {percent:6.2f}% " "({done} / {total}) at {rate}/s"
                            ).format(
                                percent=percent,
                                done=_format_size(downloaded),
                                total=_format_size(total_size),
                                rate=_format_size(int(speed)),
                            )
                            sys.stderr.write(message)
                if sys.stderr.isatty():
                    sys.stderr.write("\n")
                return destination
        except requests.RequestException as exc:
            if attempt >= max_retries:
                msg = ("Failed to download archive after {attempt} attempts: " "{error}").format(
                    attempt=attempt, error=exc
                )
                raise RuntimeError(msg) from exc
            wait_time = retry_wait * (2 ** max(0, attempt - 1))
            sys.stderr.write(
                ("Download interrupted ({error}); retrying in " "{delay:.1f}s...\n").format(
                    error=exc, delay=wait_time
                ),
            )
            time.sleep(wait_time)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    archive_path: Path | None = None
    response: requests.Response | None = None

    if args.download_to and args.archive:
        conflict_msg = "--archive and --download-to cannot be used together"
        print(conflict_msg, file=sys.stderr)
        return 1

    if args.download_to:
        archive_path = download_archive(
            args.url,
            destination=args.download_to,
            max_retries=max(args.max_download_retries, 1),
            retry_wait=max(args.retry_wait, 0.1),
        )
    elif args.archive:
        resolved_archive = args.archive.expanduser().resolve()
        if not resolved_archive.exists():
            print(f"Archive not found: {resolved_archive}", file=sys.stderr)
            return 1
        archive_path = resolved_archive
    else:
        response = open_archive_from_url(args.url)

    if archive_path is not None:
        stream = archive_path.open("rb")
    elif response is not None:
        stream = response.raw
    else:  # pragma: no cover - defensive
        raise RuntimeError("Unable to obtain archive stream")

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

    summary = ("Extraction complete. Extracted {extracted} files, " "skipped {skipped}.").format(
        **stats
    )
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
