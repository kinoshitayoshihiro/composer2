import base64
import asyncio
import io

import pytest
import pretty_midi

pytest.importorskip("pytest_asyncio")
import httpx
from httpx import AsyncClient
from fastapi.testclient import TestClient

from api import server as server_mod


@pytest.mark.asyncio
async def test_generate_endpoint():
    transport = httpx.ASGITransport(app=server_mod.app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.post("/generate", json={"model_id": "test", "chords": [60, 64], "bars": 1})
        assert resp.status_code == 200
        data = resp.json()
        midi = base64.b64decode(data["midi"])
        assert midi.startswith(b"MThd")
        pm = pretty_midi.PrettyMIDI(io.BytesIO(midi))
        assert len(pm.instruments) == 1


@pytest.mark.asyncio
async def test_websocket_broadcast():
    client = TestClient(server_mod.app)

    def run_ws():
        with client.websocket_connect("/session/demo") as ws1, client.websocket_connect("/session/demo") as ws2:
            ws1.send_json({"chords": [60], "bars": 1})
            return ws2.receive_json()

    data = await asyncio.to_thread(run_ws)
    midi = base64.b64decode(data["midi"])
    assert midi.startswith(b"MThd")
    pm = pretty_midi.PrettyMIDI(io.BytesIO(midi))
    assert len(pm.instruments) == 1
