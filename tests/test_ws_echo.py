import asyncio
import importlib.util
import json
import pytest

ws_available = importlib.util.find_spec("websockets") is not None


@pytest.mark.asyncio
@pytest.mark.skipif(not ws_available, reason="websockets missing")
async def test_ws_echo(monkeypatch):
    import websockets
    from realtime import ws_bridge

    class DummyGen:
        def __init__(self, *_a, **_k):
            pass

        def generate(self, max_bars: int = 1):
            return [1]

        def step(self, events):
            return [{"pitch": 42}]

    monkeypatch.setattr(ws_bridge, "PianoMLGenerator", DummyGen)

    model = ws_bridge.WarmModel()
    server = await ws_bridge.serve(model)
    try:
        async with websockets.connect("ws://localhost:8765") as ws:
            await ws.send(json.dumps({"chord": [60, 64, 67], "bars_context": 2}))
            resp = await ws.recv()
            data = json.loads(resp)
            assert isinstance(data, list)
    finally:
        server.close()
        await server.wait_closed()
