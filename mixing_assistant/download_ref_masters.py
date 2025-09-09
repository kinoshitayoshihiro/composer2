from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import logging
import shutil
import zipfile
from pathlib import Path

import aiohttp
from tqdm import tqdm

__all__ = ["download_refs", "main"]

logger = logging.getLogger(__name__)


async def _stream_download(
    session: aiohttp.ClientSession, url: str, dest: Path, expected_sha1: str
) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.with_suffix(".part")
    resume_pos = temp.stat().st_size if temp.exists() else 0
    headers = {"Range": f"bytes={resume_pos}-"} if resume_pos else {}
    async with session.get(url, headers=headers) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("Content-Length", 0)) + resume_pos
        sha1 = hashlib.sha1()
        if resume_pos:
            with temp.open("rb") as f:
                while chunk := f.read(8192):
                    sha1.update(chunk)
        mode = "ab" if resume_pos else "wb"
        with temp.open(mode) as f, tqdm(
            total=total,
            initial=resume_pos,
            unit="B",
            unit_scale=True,
            desc=dest.name,
        ) as pbar:
            async for chunk in resp.content.iter_chunked(65536):
                f.write(chunk)
                sha1.update(chunk)
                pbar.update(len(chunk))
    if sha1.hexdigest() != expected_sha1:
        temp.unlink(missing_ok=True)
        raise ValueError(f"Checksum mismatch for {url}")
    temp.rename(dest)


def _unzip(src: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(src) as zf:
        for member in zf.infolist():
            if ".." in Path(member.filename).parts:
                logger.warning("Skipping unsafe path %s in %s", member.filename, src)
                continue
            target = out_dir / member.filename
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as f, open(target, "wb") as o:
                shutil.copyfileobj(f, o)


async def download_refs(
    csv_path: Path, out_dir: Path, *, timeout: int = 300, concurrency: int = 4
) -> None:
    rows = list(csv.DictReader(csv_path.open()))
    sem = asyncio.Semaphore(concurrency)

    async def handle(session: aiohttp.ClientSession, row: dict[str, str]) -> None:
        url = row["url"].strip()
        sha1 = row["checksum_sha1"].strip()
        track_id = Path(url).stem
        track_dir = out_dir / track_id
        dest = track_dir / (
            "stems.zip" if url.lower().endswith(".zip") else "master.wav"
        )
        if dest.exists():
            existing = hashlib.sha1(dest.read_bytes()).hexdigest()
            if existing == sha1:
                logger.info("Skipping %s", dest)
                return
            dest.unlink()
        for attempt in range(3):
            try:
                async with sem:
                    await _stream_download(session, url, dest, sha1)
                break
            except Exception as exc:
                logger.warning("Attempt %s failed for %s: %s", attempt + 1, url, exc)
                dest.with_suffix(".part").unlink(missing_ok=True)
                if attempt == 2:
                    raise
                await asyncio.sleep(1)
        if dest.suffix == ".zip":
            _unzip(dest, track_dir / "stems")

    client_timeout = aiohttp.ClientTimeout(total=timeout)
    connector = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(
        timeout=client_timeout, connector=connector
    ) as session:
        await asyncio.gather(*(handle(session, r) for r in rows))


def main() -> None:
    parser = argparse.ArgumentParser(description="Download reference masters")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()
    asyncio.run(
        download_refs(
            args.csv, args.out, timeout=args.timeout, concurrency=args.concurrency
        )
    )


if __name__ == "__main__":
    main()
