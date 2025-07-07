import importlib

from fastapi.testclient import TestClient

from api.sax_server import app


def test_stub_generation() -> None:
    mod = importlib.import_module("plugins.sax_companion_stub")
    notes = mod.generate_notes({"growl": True})
    assert notes[0]["growl"] is True


def test_api_endpoint() -> None:
    client = TestClient(app)
    resp = client.post("/generate_sax", json={"growl": False, "altissimo": True})
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert data[0]["note"] == 72
