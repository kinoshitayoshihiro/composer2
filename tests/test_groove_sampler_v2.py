import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

pretty_midi = pytest.importorskip("pretty_midi")
mido = pytest.importorskip("mido")

import types


@pytest.fixture
def groove_sampler_v2_module(stub_utilities, monkeypatch: pytest.MonkeyPatch):
    module_path = Path(__file__).resolve().parents[1] / "utilities" / "groove_sampler_v2.py"
    stubs = {
        "loop_ingest": types.SimpleNamespace(load_meta=lambda *a, **k: {}),
        "groove_sampler": types.SimpleNamespace(
            _PITCH_TO_LABEL={},
            _iter_drum_notes=lambda *a, **k: [],
            infer_resolution=lambda *a, **k: 480,
        ),
    }
    if "joblib" not in sys.modules:
        monkeypatch.setitem(
            sys.modules,
            "joblib",
            types.SimpleNamespace(Parallel=lambda *a, **k: None, delayed=lambda f: f),
        )
    if "tqdm" not in sys.modules:
        monkeypatch.setitem(sys.modules, "tqdm", types.SimpleNamespace(tqdm=lambda x, **k: x))
    with stub_utilities(
        "utilities.groove_sampler_v2",
        module_path,
        submodules=stubs,
    ) as module:
        yield module


def _make_full_loop(path: Path) -> None:
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=0, is_drum=True)
    for i in range(16):
        inst.notes.append(
            pretty_midi.Note(velocity=100, pitch=36, start=i * 0.25, end=i * 0.25 + 0.1)
        )
    pm.instruments.append(inst)
    pm.write(str(path))


def test_train_speed_large(groove_sampler_v2_module, tmp_path: Path) -> None:
    for i in range(1000):
        _make_full_loop(tmp_path / f"{i}.mid")
    t0 = time.perf_counter()
    groove_sampler_v2_module.train(tmp_path, n=4)
    elapsed = time.perf_counter() - t0
    assert elapsed < 3.0


def test_coarse_bucket_count(groove_sampler_v2_module, tmp_path: Path) -> None:
    _make_full_loop(tmp_path / "loop.mid")
    model = groove_sampler_v2_module.train(tmp_path, coarse=True)
    buckets = {st[1] for st in model.idx_to_state}
    assert len(buckets) == model.resolution // 4


def test_sample_cli_json_sorted(groove_sampler_v2_module, tmp_path: Path) -> None:
    _make_full_loop(tmp_path / "loop.mid")
    model = groove_sampler_v2_module.train(tmp_path)
    model_path = tmp_path / "model.pkl"
    groove_sampler_v2_module.save(model, model_path)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "utilities.groove_sampler_v2",
            "sample",
            str(model_path),
            "-l",
            "1",
            "--seed",
            "0",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(result.stdout)
    offsets = [ev["offset"] for ev in data]
    assert offsets == sorted(offsets)


def test_convert_wav_auto_tempo(
    groove_sampler_v2_module, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import types

    class DummyArray(list):
        ndim = 1

    def dummy_read(path, dtype="float32"):
        return DummyArray([0.0] * 1000), 1000

    class DummyLibrosa:
        class beat:  # pragma: no cover - simple stub
            @staticmethod
            def beat_track(y, sr, trim=False):
                return 128.0, []

    monkeypatch.setitem(sys.modules, "soundfile", types.SimpleNamespace(read=dummy_read))
    monkeypatch.setitem(sys.modules, "librosa", DummyLibrosa())

    pm = groove_sampler_v2_module.convert_wav_to_midi(tmp_path / "loop.wav")
    assert pm is not None
    tempi, _times = pm.get_tempo_changes()
    assert pytest.approx(tempi[0], abs=1e-6) == 128.0


def test_convert_wav_tempo_fallback(
    groove_sampler_v2_module, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import types
    import builtins

    class DummyArray(list):
        ndim = 1

    def dummy_read(path, dtype="float32"):
        return DummyArray([0.0] * 1000), 1000

    monkeypatch.setitem(sys.modules, "soundfile", types.SimpleNamespace(read=dummy_read))

    orig_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "librosa":
            raise ImportError("missing")
        return orig_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    pm = groove_sampler_v2_module.convert_wav_to_midi(tmp_path / "loop.wav")
    assert pm is not None
    tempi, _times = pm.get_tempo_changes()
    assert pytest.approx(tempi[0], abs=1e-6) == 120.0


def test_convert_wav_fixed_bpm(
    groove_sampler_v2_module, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import types

    class DummyArray(list):
        ndim = 1

    def dummy_read(path, dtype="float32"):
        return DummyArray([0.0] * 1000), 1000
    monkeypatch.setitem(sys.modules, "soundfile", types.SimpleNamespace(read=dummy_read))
    pm = groove_sampler_v2_module.convert_wav_to_midi(
        tmp_path / "loop.wav", fixed_bpm=90
    )
    assert pm is not None
    tempi, _times = pm.get_tempo_changes()
    assert pytest.approx(tempi[0], abs=1e-6) == 90.0

