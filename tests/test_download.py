import asyncio
import csv
import hashlib
import io
import sys
import types
import zipfile
from pathlib import Path

import pytest

sys.modules.setdefault(
    "mixing_assistant.feature_extractor",
    types.ModuleType("mixing_assistant.feature_extractor"),
)
sys.modules["mixing_assistant.feature_extractor"].extract_features = (
    lambda *_a, **_k: None
)
import mixing_assistant.download_ref_masters as drm  # noqa: E402


class _Resp:
    def __init__(self, data: bytes) -> None:
        self._data = data
        self.headers = {"Content-Length": str(len(data))}

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass

    def raise_for_status(self) -> None:
        pass

    @property
    def content(self):
        return self

    async def iter_chunked(self, n: int):
        yield self._data


class _Session:
    data: list[bytes]
    calls = 0

    def __init__(self, *a, **k) -> None:
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass

    def get(self, url: str, headers=None):
        data = self.data[self.calls]
        _Session.calls += 1
        return _Resp(data)


def test_download_retry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bad = io.BytesIO()
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("f.txt", b"bad")
    good = io.BytesIO()
    with zipfile.ZipFile(good, "w") as zf:
        zf.writestr("f.txt", b"good")
    _Session.data = [bad.getvalue(), good.getvalue()]
    sha1 = hashlib.sha1(_Session.data[1]).hexdigest()
    csv_path = tmp_path / "files.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["url", "checksum_sha1"])
        writer.writeheader()
        writer.writerow({"url": "http://x/f.zip", "checksum_sha1": sha1})
    out_dir = tmp_path / "out"
    monkeypatch.setattr(
        drm,
        "aiohttp",
        types.SimpleNamespace(
            ClientSession=_Session,
            ClientTimeout=lambda total: None,
            TCPConnector=lambda limit=None: None,
        ),
    )
    asyncio.run(drm.download_refs(csv_path, out_dir, timeout=5, concurrency=1))
    assert _Session.calls == 2
    assert (out_dir / "f" / "stems" / "f.txt").read_bytes() == b"good"
