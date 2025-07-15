import asyncio
import importlib.util
import json
import statistics
import time

import pytest

ws_available = importlib.util.find_spec("websockets") is not None


@pytest.mark.asyncio
@pytest.mark.skipif(not ws_available, reason="websockets missing")
async def test_ws_latency():
    import websockets
    from utilities import ws_bridge

    server = await ws_bridge.serve(port=8765)
    try:
        async with websockets.connect("ws://localhost:8765/groove") as ws:
            lat: list[float] = []
            for i in range(1000):
                payload = {"type": "note", "i": i}
                start = time.perf_counter()
                await ws.send(json.dumps(payload))
                resp = await ws.recv()
                assert json.loads(resp)["i"] == i
                lat.append((time.perf_counter() - start) * 1000.0)
            p99 = statistics.quantiles(lat, n=100)[98]
            assert p99 <= 20.0
    finally:
        server.close()
        await server.wait_closed()
