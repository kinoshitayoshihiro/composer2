import importlib
from fastapi.testclient import TestClient

import types
import sys
import pytest


def test_stub_generation() -> None:
    mod = importlib.import_module("plugins.sax_companion_stub")
    notes = mod.generate_notes({"growl": True})
    assert notes[0]["growl"] is True


def test_api_endpoint() -> None:
    from api.sax_server import app
    client = TestClient(app)
    resp = client.post("/generate_sax", json={"growl": False, "altissimo": True})
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert data[0]["note"] == 72


def test_validation_error() -> None:
    from api.sax_server import app
    client = TestClient(app)
    resp = client.post("/generate_sax", json={"growl": True, "unknown": 1})
    assert resp.status_code == 422


def test_plugin_failure(monkeypatch) -> None:
    def bad_generate(_):
        return [{"error": "StubError", "message": "boom"}]

    fake_mod = types.ModuleType("plugins.sax_companion_stub")
    fake_mod.generate_notes = bad_generate
    monkeypatch.setitem(sys.modules, "plugins.sax_companion_stub", fake_mod)
    import plugins
    monkeypatch.setattr(plugins, "sax_companion_stub", fake_mod, raising=False)
    import importlib
    sys.modules.pop("api.sax_server", None)
    import api.sax_server as sax_server
    client = TestClient(sax_server.app)
    resp = client.post("/generate_sax", json={"growl": True})
    assert resp.status_code == 500
    assert resp.json()["detail"] == "boom"
