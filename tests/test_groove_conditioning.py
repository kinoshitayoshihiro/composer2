from pathlib import Path
import importlib.machinery
import importlib.util
import sys
import types
import json
import subprocess
import os
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


def test_cond_kick_four_on_floor() -> None:
    model = make_model(["snare"])
    events = groove_sampler_v2.generate_events(
        model, bars=2, seed=0, cond_kick="four_on_floor", cond={"feel": "laidback"}
    )
    beats = 2 * 4
    for b in range(beats):
        start = float(b)
        kicks = [
            ev
            for ev in events
            if ev["instrument"] == "kick" and start <= ev["offset"] < start + 1
        ]
        assert len(kicks) == 1


def test_cond_velocity_soft() -> None:
    model = make_model(["snare"])
    events_base = groove_sampler_v2.generate_events(model, bars=1, seed=0)
    events_soft = groove_sampler_v2.generate_events(
        model, bars=1, seed=0, cond_velocity="soft"
    )

    def low_ratio(evts):
        return sum(ev["velocity_factor"] < 1.0 for ev in evts) / len(evts)

    assert low_ratio(events_soft) > low_ratio(events_base)


def test_ohh_choke_prob() -> None:
    model = make_model(["ohh"])
    events_yes = groove_sampler_v2.generate_events(
        model, bars=1, seed=0, ohh_choke_prob=1.0
    )
    assert any(ev["instrument"] == "hh_pedal" for ev in events_yes)
    events_no = groove_sampler_v2.generate_events(
        model, bars=1, seed=0, ohh_choke_prob=0.0
    )
    assert all(ev["instrument"] != "hh_pedal" for ev in events_no)


def test_ohh_choke_conflict() -> None:
    model = make_model(["ohh"])
    # force ohh on every tick
    for i in range(model.resolution):
        arr = np.zeros(len(model.idx_to_state))
        arr[0] = 1.0
        model.bucket_freq[i] = arr
    events = groove_sampler_v2.generate_events(
        model, bars=1, seed=0, ohh_choke_prob=1.0
    )
    ohhs = [e for e in events if e["instrument"] == "hh_open"]
    pedals = [e for e in events if e["instrument"] == "hh_pedal"]
    assert len(pedals) == len(ohhs)
    offsets = set()
    for o in ohhs:
        expected = o["offset"] + 1.0
        matches = [p for p in pedals if abs(p["offset"] - expected) < 1e-6]
        assert len(matches) == 1
        offsets.add(matches[0]["offset"])
    assert len(offsets) == len(pedals)


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
