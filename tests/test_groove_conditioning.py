import importlib.machinery
import importlib.util
import json
import os
import subprocess
import sys
import types
from pathlib import Path

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


def test_determinism() -> None:
    model = make_model(["snare"])
    groove_sampler_v2.set_random_state(0)
    ev1 = groove_sampler_v2.generate_events(model, bars=1)
    groove_sampler_v2.set_random_state(0)
    ev2 = groove_sampler_v2.generate_events(model, bars=1)
    assert [e["instrument"] for e in ev1] == [e["instrument"] for e in ev2]


def test_topk_one() -> None:
    model = make_model(["kick", "snare"])
    for i in range(model.resolution):
        model.bucket_freq[i] = np.array([10.0, 1.0])
    groove_sampler_v2.set_random_state(0)
    events = groove_sampler_v2.generate_events(model, bars=1, top_k=1)
    assert all(ev["instrument"] == "kick" for ev in events)


def test_length_max_steps() -> None:
    model = make_model(["snare"])
    groove_sampler_v2.set_random_state(0)
    events = groove_sampler_v2.generate_events(model, bars=4, max_steps=5)
    assert len(events) <= 5
    assert max(e["offset"] for e in events) < 16


def test_conditioning_bias() -> None:
    model = make_model(["kick", "snare"])
    aux_id = model.aux_vocab.encode({"style": "funk"})
    ctx_hash = groove_sampler_v2.hash_ctx([0, aux_id]) % model.hash_buckets
    model.freq[0][ctx_hash] = np.array([10.0, 1.0])
    probs = groove_sampler_v2.next_prob_dist(model, [], 0, cond={"style": "funk"})
    assert probs[0] > probs[1]


def test_ohh_choke_prob() -> None:
    model = make_model(["hh_open"])
    groove_sampler_v2.set_random_state(0)
    events_yes = groove_sampler_v2.generate_events(model, bars=1, ohh_choke_prob=1.0)
    assert any(ev["instrument"] == "hh_pedal" for ev in events_yes)
    groove_sampler_v2.set_random_state(0)
    events_no = groove_sampler_v2.generate_events(model, bars=1, ohh_choke_prob=0.0)
    assert all(ev["instrument"] != "hh_pedal" for ev in events_no)


def test_cli_smoke(tmp_path: Path) -> None:
    model = make_model(["snare"])
    model_path = tmp_path / "dummy.pkl"
    groove_sampler_v2.save(model, model_path)
    env = {**os.environ, "PYTHONPATH": str(UTILS_PATH.parent)}
    cmd = [
        sys.executable,
        "-m",
        "utilities.groove_sampler_v2",
        "sample",
        str(model_path),
        "-l",
        "1",
        "--print-json",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=tmp_path, env=env)
    assert proc.returncode == 0
    json.loads(proc.stdout)
    assert not list(tmp_path.glob("*.mid"))

    out_mid = tmp_path / "out.mid"
    cmd2 = [
        sys.executable,
        "-m",
        "utilities.groove_sampler_v2",
        "sample",
        str(model_path),
        "-l",
        "1",
        "--out-midi",
        str(out_mid),
    ]
    proc2 = subprocess.run(cmd2, capture_output=True, text=True, cwd=tmp_path, env=env)
    assert proc2.returncode == 0
    assert out_mid.exists()
