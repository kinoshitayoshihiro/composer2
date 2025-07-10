import base64
import asyncio
import io

import pytest

pytest.importorskip("pytest_asyncio")
import httpx
from httpx import AsyncClient
from fastapi.testclient import TestClient

from api import vocal_server as server
import pretty_midi


@pytest.mark.asyncio
async def test_generate_vocal_endpoint():
    transport = httpx.ASGITransport(app=server.app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.post(
            "/generate_vocal",
            json={"model_id": "test", "chords": [60, 64], "bars": 1},
        )
        assert resp.status_code == 200
        data = resp.json()
        midi = base64.b64decode(data["midi"])
        assert midi.startswith(b"MThd")
        pm = pretty_midi.PrettyMIDI(io.BytesIO(midi))
        assert len(pm.instruments) == 1


@pytest.mark.asyncio
async def test_ws_vocal_session_broadcast():
    client = TestClient(server.app)

    def run_ws():
        with client.websocket_connect("/session/vocal/demo") as ws1, client.websocket_connect("/session/vocal/demo") as ws2:
            ws1.send_json({"chords": [60], "bars": 1})
            return ws2.receive_json()

    data = await asyncio.to_thread(run_ws)
    midi = base64.b64decode(data["midi"])
    assert midi.startswith(b"MThd")
    pm = pretty_midi.PrettyMIDI(io.BytesIO(midi))
    assert len(pm.instruments) == 1
