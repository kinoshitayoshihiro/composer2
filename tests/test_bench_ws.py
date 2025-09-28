import importlib.machinery
import importlib.util
import sys
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from types import ModuleType

import pytest
from pytest import MonkeyPatch

pytestmark = pytest.mark.asyncio


class DummyWS:
    async def send(self, data: bytes) -> None:  # pragma: no cover - stub
        pass

    async def recv(self) -> bytes:
        return b"[]"


class DummyServeHandle:
    async def wait_closed(self) -> None:  # pragma: no cover - stub
        return None


def _load(patch: MonkeyPatch) -> ModuleType:
    ws_mod = ModuleType("websockets")
    ws_mod.__spec__ = importlib.machinery.ModuleSpec("websockets", loader=None)

    @asynccontextmanager
    async def _connect(*_args: object, **_kwargs: object):
        yield DummyWS()

    async def _serve(*_args: object, **_kwargs: object) -> DummyServeHandle:
        return DummyServeHandle()

    setattr(ws_mod, "connect", _connect)
    setattr(ws_mod, "serve", _serve)

    ws_asyncio = ModuleType("websockets.asyncio")
    ws_asyncio.__spec__ = importlib.machinery.ModuleSpec("websockets.asyncio", loader=None)
    setattr(ws_asyncio, "connect", _connect)
    setattr(ws_asyncio, "serve", _serve)

    patch.setitem(sys.modules, "websockets", ws_mod)
    patch.setitem(sys.modules, "websockets.asyncio", ws_asyncio)

    fastapi = ModuleType("fastapi")
    fastapi.__spec__ = importlib.machinery.ModuleSpec("fastapi", loader=None)

    class DummyApp:
        def post(self, *_a: object, **_k: object) -> Callable[[object], object]:
            def decorator(fn: object) -> object:
                return fn

            return decorator

        def get(self, *_a: object, **_k: object) -> Callable[[object], object]:
            def decorator(fn: object) -> object:
                return fn

            return decorator

        def websocket(self, *_a: object, **_k: object) -> Callable[[object], object]:
            def decorator(fn: object) -> object:
                return fn

            return decorator

        async def __call__(self, scope: object, receive: object, send: object) -> None:
            return None  # pragma: no cover - stub

    setattr(fastapi, "FastAPI", lambda: DummyApp())
    setattr(fastapi, "WebSocket", object)
    patch.setitem(sys.modules, "fastapi", fastapi)

    uvicorn = ModuleType("uvicorn")
    uvicorn.__spec__ = importlib.machinery.ModuleSpec("uvicorn", loader=None)

    class Server:
        def __init__(self, config: object) -> None:  # pragma: no cover - stub
            self.should_exit = False
            self.config = config

        async def serve(self) -> None:  # pragma: no cover - stub
            return None

    def _config(*_args: object, **_kwargs: object) -> object:
        return object()

    setattr(uvicorn, "Config", _config)
    setattr(uvicorn, "Server", Server)
    patch.setitem(sys.modules, "uvicorn", uvicorn)

    ymod = ModuleType("yaml")
    ymod.__spec__ = importlib.util.spec_from_loader("yaml", loader=None)
    patch.setitem(sys.modules, "yaml", ymod)

    spec = importlib.util.spec_from_file_location(
        "scripts.bench_ws",
        Path(__file__).resolve().parents[1] / "scripts" / "bench_ws.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    patch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.asyncio
async def test_bench_ws(monkeypatch: MonkeyPatch) -> None:
    with monkeypatch.context() as patch:
        mod = _load(patch)
        latency = await mod.bench(n=2)

    assert latency < 50.0
