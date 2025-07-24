import asyncio
import importlib.util
import sys
import time
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.asyncio


def _load_module() -> ModuleType:
    if importlib.util.find_spec("fastapi") is None:
        fastapi = ModuleType("fastapi")

        class DummyApp:
            def post(self, *_a, **_k):
                return lambda fn: fn

            websocket = post

        fastapi.FastAPI = lambda: DummyApp()
        fastapi.WebSocket = object
        fastapi.__spec__ = importlib.machinery.ModuleSpec("fastapi", loader=None)
        sys.modules["fastapi"] = fastapi
    if importlib.util.find_spec("uvicorn") is None:
        uvicorn = ModuleType("uvicorn")

        class Server:
            def __init__(self, config: object) -> None:  # pragma: no cover - stub
                pass

            async def serve(self) -> None:  # pragma: no cover - stub
                pass

        uvicorn.Config = lambda *a, **_k: object()
        uvicorn.Server = Server
        uvicorn.__spec__ = importlib.machinery.ModuleSpec("uvicorn", loader=None)
        sys.modules["uvicorn"] = uvicorn
    spec = importlib.util.spec_from_file_location(
        "phrase_ws", Path(__file__).resolve().parents[1] / "realtime" / "phrase_ws.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_midi() -> bytes:
    """Return a tiny valid MIDI file."""
    return bytes.fromhex("4d54686400000006000100010060" "4d54726b0000000400ff2f00")


class DummyWebSocket:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.sent: list | None = None

    async def accept(self) -> None:  # pragma: no cover - stub
        pass

    async def receive_bytes(self) -> bytes:
        if self.payload is None:
            raise RuntimeError("done")
        d = self.payload
        self.payload = None
        return d

    async def send_json(self, obj: list) -> None:
        self.sent = obj
        raise RuntimeError("done")


@pytest.mark.asyncio
async def test_realtime_ws() -> None:
    mod = _load_module()

    def dummy_feats(_: bytes):
        import torch

        feats = {
            "pitch_class": torch.tensor([[0, 1]], dtype=torch.long),
            "velocity": torch.tensor([[64.0, 64.0]]),
            "duration": torch.tensor([[1.0, 1.0]]),
            "position": torch.tensor([[0, 1]], dtype=torch.long),
        }
        mask = torch.ones(1, 2, dtype=torch.bool)
        return feats, mask

    mod.segment_bytes.__globals__["_midi_to_feats"] = dummy_feats

    class FakeModel:
        def __call__(self, feats: object, mask: object) -> list:
            import torch

            return [torch.tensor([0.0, 2.0])]

    mod._model = FakeModel()
    ws = DummyWebSocket(_make_midi())
    start = time.perf_counter()
    with pytest.raises(RuntimeError):
        await mod.infer(ws)
    latency = (time.perf_counter() - start) * 1000.0
    assert latency < 50.0
    assert isinstance(ws.sent, list)
    assert ws.sent and isinstance(ws.sent[0][0], int)
    assert isinstance(ws.sent[0][1], float)


@pytest.mark.asyncio
async def test_warmup_concurrent() -> None:
    mod = _load_module()
    calls = 0

    def fake_load_model(*_: object) -> object:
        nonlocal calls
        calls += 1
        time.sleep(0.01)
        return object()

    mod.load_model = fake_load_model  # type: ignore[attr-defined]
    res = await asyncio.gather(mod.warmup(), mod.warmup())
    assert calls == 1
    assert {r["status"] for r in res} == {"ok", "busy"}


def test_transformer_pointer_shape() -> None:
    import torch

    from models.phrase_transformer import PhraseTransformer

    model = PhraseTransformer(d_model=8, max_len=4)
    feats = {
        "pitch_class": torch.zeros(1, 2, dtype=torch.long),
        "velocity": torch.zeros(1, 2),
        "duration": torch.zeros(1, 2),
        "position": torch.zeros(1, 2, dtype=torch.long),
    }
    mask = torch.ones(1, 2, dtype=torch.bool)
    out = model(feats, mask)
    assert out.shape == (1, 2)
