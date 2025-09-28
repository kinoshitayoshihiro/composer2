import json
from pathlib import Path

import pytest

try:  # pragma: no cover - pretty_midi optional
    import pretty_midi  # type: ignore
except Exception:  # pragma: no cover
    from ._stubs import pretty_midi  # noqa: F401

import sys
import types

import utilities.groove_sampler_v2 as groove_sampler_v2


@pytest.fixture(autouse=True)
def _stub_loop_ingest(monkeypatch):
    stub = types.ModuleType("utilities.loop_ingest")
    stub.load_meta = lambda path: {}  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "utilities.loop_ingest", stub)
    monkeypatch.setattr(
        groove_sampler_v2,
        "load_meta",
        stub.load_meta,
        raising=False,
    )
    monkeypatch.setattr(
        "utilities.groove_sampler_v2._LOAD_META_FALLBACK",
        stub.load_meta,
        raising=False,
    )
    groove_sampler_v2.set_load_meta_override(None)
    try:
        yield
    finally:
        groove_sampler_v2.set_load_meta_override(None)


def test_aux_extraction_precedence(tmp_path):
    mid = tmp_path / "song__mood-happy__section-verse.mid"
    mid.touch()
    yaml_path = tmp_path / "aux.yaml"
    yaml_path.write_text("song__mood-happy__section-verse.mid:\n  mood: sad\n")

    mapping = json.loads(json.dumps({mid.name: {"mood": "sad"}}))
    assert groove_sampler_v2._extract_aux(
        mid,
        aux_map=mapping,
        aux_key="mood",
    ) == {"mood": "sad"}

    def fake_meta(path):
        return {"mood": "meta"}

    groove_sampler_v2.set_load_meta_override(fake_meta)
    assert groove_sampler_v2._extract_aux(mid, aux_key="mood") == {"mood": "meta"}
    assert groove_sampler_v2._extract_aux(mid) == {
        "mood": "happy",
        "section": "verse",
    }


def test_aux_extraction_blank_metadata_falls_back(tmp_path):
    mid = tmp_path / "song__mood-happy__section-verse.mid"
    mid.touch()

    def fake_meta(_path):
        return {"mood": "  "}

    groove_sampler_v2.set_load_meta_override(fake_meta)
    expected = {"mood": "happy", "section": "verse"}
    assert groove_sampler_v2._extract_aux(mid, aux_key="mood") == expected
