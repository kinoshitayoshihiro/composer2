import importlib.util
import platform
import sys
import types
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pretty_midi = pytest.importorskip("pretty_midi")
psutil = pytest.importorskip("psutil")

_spec = importlib.util.spec_from_file_location(
    "utilities.groove_sampler_v2",
    Path(__file__).resolve().parents[1] / "utilities" / "groove_sampler_v2.py",
)
assert _spec.loader is not None
module = importlib.util.module_from_spec(_spec)
pkg = types.ModuleType("utilities")
pkg.loop_ingest = types.SimpleNamespace(load_meta=lambda *a, **k: {})
pkg.pretty_midi_safe = types.SimpleNamespace(pm_to_mido=lambda pm: pm)
pkg.aux_vocab = types.SimpleNamespace(AuxVocab=object)
pkg.groove_sampler = types.SimpleNamespace(
    _PITCH_TO_LABEL={}, infer_resolution=lambda *a, **k: 16
)
sys.modules["utilities"] = pkg
sys.modules["utilities.loop_ingest"] = pkg.loop_ingest
sys.modules["utilities.pretty_midi_safe"] = pkg.pretty_midi_safe
sys.modules["utilities.aux_vocab"] = pkg.aux_vocab
sys.modules["utilities.groove_sampler"] = pkg.groove_sampler
_spec.loader.exec_module(module)


def _make_loop(path: Path, notes: int) -> None:
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=0, is_drum=True)
    for i in range(notes):
        inst.notes.append(
            pretty_midi.Note(velocity=100, pitch=36, start=i * 0.25, end=i * 0.25 + 0.1)
        )
    pm.instruments.append(inst)
    pm.write(str(path))


def test_hash_deterministic() -> None:
    h = module._hash_ctx([1, 2, 3])
    assert h == 11402386789778234582


def test_memmap_memory_and_probs(tmp_path: Path) -> None:
    for i in range(5):
        _make_loop(tmp_path / f"loop{i}.mid", 32)
    mem_dir = tmp_path / "mm"
    model_mem = module.train(
        tmp_path,
        memmap_dir=mem_dir,
        snapshot_interval=1,
        counts_dtype="uint16",
        hash_buckets=256,
    )
    model_ram = module.train(tmp_path, hash_buckets=256)
    hb = model_ram.hash_buckets
    h = module._hash_ctx([0, 0]) % hb
    arr1 = model_ram.freq[0][h]
    arr2 = model_mem.freq[0][h]
    p1 = arr1 / arr1.sum()
    p2 = arr2 / arr2.sum()
    assert np.allclose(p1, p2)
    if platform.system() == "Linux":
        rss = psutil.Process().memory_info().rss / (1024 ** 2)
        assert rss < 300
