import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.asyncio


class DummyWS:
    async def __aenter__(self):  # pragma: no cover - stub
        return self

    async def __aexit__(self, *exc):  # pragma: no cover - stub
        pass

    async def send(self, data: bytes) -> None:  # pragma: no cover - stub
        pass

    async def recv(self) -> bytes:
        return b"[]"


def _load() -> ModuleType:
    ws_mod = ModuleType("websockets")
    ws_mod.connect = lambda *a, **k: DummyWS()
    sys.modules["websockets"] = ws_mod

    fastapi = ModuleType("fastapi")

    class DummyApp:
        def post(self, *_a, **_k):
            return lambda fn: fn

        websocket = post

    fastapi.FastAPI = lambda: DummyApp()
    fastapi.WebSocket = object
    sys.modules["fastapi"] = fastapi

    uvicorn = ModuleType("uvicorn")

    class Server:
        def __init__(self, config: object) -> None:  # pragma: no cover - stub
            self.should_exit = False

        async def serve(self) -> None:  # pragma: no cover - stub
            pass

    uvicorn.Config = lambda *a, **k: object()
    uvicorn.Server = Server
    sys.modules["uvicorn"] = uvicorn

    ymod = ModuleType("yaml")
    ymod.__spec__ = importlib.util.spec_from_loader("yaml", loader=None)
    sys.modules.setdefault("yaml", ymod)

    spec = importlib.util.spec_from_file_location(
        "bench_ws", Path(__file__).resolve().parents[1] / "scripts" / "bench_ws.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.asyncio
async def test_bench_ws() -> None:
    mod = _load()
    latency = await mod.bench(n=2)
    assert latency < 50.0
