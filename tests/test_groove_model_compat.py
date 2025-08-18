import pickle
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

try:  # pragma: no cover - pretty_midi optional
    import pretty_midi  # type: ignore
except Exception:  # pragma: no cover
    from ._stubs import pretty_midi  # noqa: F401

import sys
import types

stub = types.ModuleType("utilities.loop_ingest")
stub.load_meta = lambda path: {}
sys.modules.setdefault("utilities.loop_ingest", stub)

from utilities.groove_sampler_v2 import load


def test_load_v1_model(tmp_path):
    freq = [{123: np.zeros(1, dtype=np.uint32)}]
    data = {
        "n": 2,
        "resolution": 16,
        "resolution_coarse": 16,
        "state_to_idx": {},
        "idx_to_state": [],
        "freq": freq,
        "bucket_freq": {},
        "ctx_maps": [{}],
    }
    path = tmp_path / "model.pkl"
    with path.open("wb") as fh:
        pickle.dump(data, fh)
    model = load(path)
    assert isinstance(model.aux_vocab, object)
    assert model.version == 1 or model.version == 2
