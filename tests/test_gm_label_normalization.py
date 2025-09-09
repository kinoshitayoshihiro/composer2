from pathlib import Path
import importlib.machinery
import importlib.util
import sys
import types
import pytest

np = pytest.importorskip("numpy")

UTILS_PATH = Path(__file__).resolve().parents[1] / "utilities"
pkg = types.ModuleType("utilities")
pkg.__path__ = [str(UTILS_PATH)]
sys.modules.setdefault("utilities", pkg)
loader = importlib.machinery.SourceFileLoader(
    "utilities.groove_sampler_v2", str(UTILS_PATH / "groove_sampler_v2.py")
)
spec = importlib.util.spec_from_loader(loader.name, loader)
groove_sampler_v2 = importlib.util.module_from_spec(spec)
sys.modules[loader.name] = groove_sampler_v2
loader.exec_module(groove_sampler_v2)

loader_map = importlib.machinery.SourceFileLoader(
    "utilities.gm_perc_map", str(UTILS_PATH / "gm_perc_map.py")
)
spec_map = importlib.util.spec_from_loader(loader_map.name, loader_map)
gm_perc_map = importlib.util.module_from_spec(spec_map)
sys.modules[spec_map.name] = gm_perc_map
loader_map.exec_module(gm_perc_map)
NAME_TO_NUM = gm_perc_map.NAME_TO_NUM


def make_model(labels: list[str]) -> groove_sampler_v2.NGramModel:
    res = 4
    idx_to_state = []
    state_to_idx = {}
    for i, lbl in enumerate(labels):
        state = (0, i % res, lbl)
        idx_to_state.append(state)
        state_to_idx[state] = i
    freq = [dict()]
    bucket_freq = {i: np.ones(len(labels)) for i in range(res)}
    ctx_maps = [{}]
    return groove_sampler_v2.NGramModel(
        n=1,
        resolution=res,
        resolution_coarse=res,
        state_to_idx=state_to_idx,
        idx_to_state=idx_to_state,
        freq=freq,
        bucket_freq=bucket_freq,
        ctx_maps=ctx_maps,
        prob=None,
        aux_vocab=groove_sampler_v2.AuxVocab(),
        hash_buckets=1,
    )


def test_instrument_normalization() -> None:
    model = make_model(["37", "ohh", "closed_hat"])
    # deterministically cycle through the provided labels
    for i, arr in model.bucket_freq.items():
        arr[:] = 0
    model.bucket_freq[0][0] = 1  # 37 -> sidestick
    model.bucket_freq[1][1] = 1  # ohh -> hh_open
    model.bucket_freq[2][2] = 1  # closed_hat -> hh_closed
    model.bucket_freq[3][0] = 1
    events = groove_sampler_v2.generate_events(model, bars=1, seed=0)
    names = {ev["instrument"] for ev in events}
    assert "37" not in names and "ohh" not in names and "closed_hat" not in names
    assert "sidestick" in names and "hh_open" in names and "hh_closed" in names
    for inst in names:
        assert inst in NAME_TO_NUM or inst.startswith("unk_")
