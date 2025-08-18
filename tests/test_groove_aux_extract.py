import json
from pathlib import Path

import pytest

try:  # pragma: no cover - pretty_midi optional
    import pretty_midi  # type: ignore
except Exception:  # pragma: no cover
    from ._stubs import pretty_midi  # noqa: F401

import sys
import types

stub = types.ModuleType("utilities.loop_ingest")
stub.load_meta = lambda path: {}
sys.modules.setdefault("utilities.loop_ingest", stub)

from utilities.groove_sampler_v2 import _extract_aux


def test_aux_extraction_precedence(tmp_path, monkeypatch):
    mid = tmp_path / "song__mood-happy__section-verse.mid"
    mid.touch()
    yaml_path = tmp_path / "aux.yaml"
    yaml_path.write_text("song__mood-happy__section-verse.mid:\n  mood: sad\n")

    mapping = json.loads(json.dumps({mid.name: {"mood": "sad"}}))
    assert _extract_aux(mid, aux_map=mapping, aux_key="mood") == {"mood": "sad"}

    def fake_meta(path):
        return {"mood": "meta"}

    monkeypatch.setattr("utilities.groove_sampler_v2.load_meta", fake_meta)
    assert _extract_aux(mid, aux_key="mood") == {"mood": "meta"}
    assert _extract_aux(mid) == {"mood": "happy", "section": "verse"}
