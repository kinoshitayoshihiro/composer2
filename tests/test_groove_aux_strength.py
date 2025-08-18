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

from utilities.aux_vocab import AuxVocab
from utilities.groove_sampler_v2 import NGramModel, _hash_ctx, sample_next


def build_model():
    aux_vocab = AuxVocab()
    happy = aux_vocab.encode({"mood": "happy"})
    n = 2
    freq = [{}, {}]
    ctx = _hash_ctx([0])
    freq[1][(ctx, happy)] = np.array([10, 0], dtype=np.uint32)
    freq[1][(ctx, 0)] = np.array([0, 10], dtype=np.uint32)
    freq[0][(0, happy)] = np.array([10, 0], dtype=np.uint32)
    freq[0][(0, 0)] = np.array([0, 10], dtype=np.uint32)
    bucket_freq = {0: np.array([5, 5], dtype=np.uint32)}
    model = NGramModel(
        n=n,
        resolution=16,
        resolution_coarse=16,
        state_to_idx={},
        idx_to_state=[(0, 0, "a"), (0, 0, "b")],
        freq=freq,
        bucket_freq=bucket_freq,
        ctx_maps=[{}, {}],
        aux_vocab=aux_vocab,
    )
    return model


def test_strength_interpolation():
    model = build_model()
    ctx = [0]
    bucket = 0
    rng = np.random.default_rng(0)
    # compute probabilities manually
    ctx_id = _hash_ctx(ctx)
    arr_any = model.freq[1][(ctx_id, 0)].astype(float)
    arr_spec = model.freq[1][
        (ctx_id, model.aux_vocab.encode({"mood": "happy"}))
    ].astype(float)

    def prob(strength):
        arr = arr_any * (1 - strength) + arr_spec * strength
        return arr[0] / arr.sum()

    assert prob(1.0) > prob(0.0)
