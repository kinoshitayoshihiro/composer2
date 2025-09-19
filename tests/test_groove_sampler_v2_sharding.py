from pathlib import Path
import types

import pytest

pretty_midi = pytest.importorskip("pretty_midi")
np = pytest.importorskip("numpy")

@pytest.fixture
def groove_sampler_v2_module(stub_utilities):
    module_path = Path(__file__).resolve().parents[1] / "utilities" / "groove_sampler_v2.py"
    stubs = {
        "loop_ingest": types.SimpleNamespace(load_meta=lambda *a, **k: {}),
        "pretty_midi_safe": types.SimpleNamespace(pm_to_mido=lambda pm: pm),
        "aux_vocab": types.SimpleNamespace(AuxVocab=object),
        "groove_sampler": types.SimpleNamespace(
            _PITCH_TO_LABEL={}, infer_resolution=lambda *a, **k: 480
        ),
    }
    with stub_utilities(
        "utilities.groove_sampler_v2",
        module_path,
        submodules=stubs,
    ) as module:
        yield module


def _make_loop(path: Path, notes: int) -> None:
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=0, is_drum=True)
    for i in range(notes):
        inst.notes.append(
            pretty_midi.Note(velocity=100, pitch=36, start=i * 0.25, end=i * 0.25 + 0.1)
        )
    pm.instruments.append(inst)
    pm.write(str(path))


def test_shard_and_merge(groove_sampler_v2_module, tmp_path: Path) -> None:
    good1 = tmp_path / "good1.mid"
    bad1 = tmp_path / "bad1.mid"
    bad2 = tmp_path / "bad2.mid"
    good2 = tmp_path / "good2.mid"
    _make_loop(good1, 16)
    _make_loop(bad1, 8)  # min-bytes filter (size <150)
    _make_loop(bad2, 4)  # min-notes filter
    _make_loop(good2, 16)
    paths = [good1, bad1, bad2, good2]

    shard0 = [p for i, p in enumerate(paths) if i % 2 == 0]
    shard1 = [p for i, p in enumerate(paths) if i % 2 == 1]

    out0 = tmp_path / "part0.pkl"
    ckpt0 = tmp_path / "ckpt0"
    res0 = groove_sampler_v2_module.train_streaming(
        shard0,
        output=out0,
        min_bytes=150,
        min_notes=8,
        save_every=1,
        checkpoint_dir=ckpt0,
        log_every=1,
        progress=True,
    )
    assert (ckpt0 / "ckpt_1.pkl").exists()
    assert res0["kept"] == 1
    assert res0["counts"][36] == 16

    out1 = tmp_path / "part1.pkl"
    ckpt1 = tmp_path / "ckpt1"
    res1 = groove_sampler_v2_module.train_streaming(
        shard1,
        output=out1,
        min_bytes=150,
        min_notes=8,
        save_every=1,
        checkpoint_dir=ckpt1,
        log_every=1,
        progress=True,
    )
    assert (ckpt1 / "ckpt_1.pkl").exists()
    assert res1["kept"] == 1
    assert res1["counts"][36] == 16

    merged = groove_sampler_v2_module.merge_streaming_models(
        [out0, out1], tmp_path / "merged.pkl"
    )
    assert merged["kept"] == 2
    assert merged["counts"][36] == 32


def test_train_modes_equivalent_and_aux_vocab(
    groove_sampler_v2_module, tmp_path: Path
) -> None:
    for i in range(3):
        _make_loop(tmp_path / f"loop{i}.mid", 16)
    aux_path = tmp_path / "aux.json"
    model_stream = groove_sampler_v2_module.train(
        tmp_path,
        memmap_dir=tmp_path / "mm",
        hash_buckets=256,
        train_mode="stream",
        aux_vocab_path=aux_path,
    )
    model_mem = groove_sampler_v2_module.train(
        tmp_path, hash_buckets=256, train_mode="inmemory"
    )
    hb = model_mem.hash_buckets
    h = groove_sampler_v2_module._hash_ctx([0, 0]) % hb
    arr1 = model_mem.freq[0][h]
    arr2 = model_stream.freq[0][h]
    p1 = arr1 / arr1.sum()
    p2 = arr2 / arr2.sum()
    assert np.allclose(p1, p2)
    assert aux_path.exists()


def test_streaming_resume_checkpoint(
    groove_sampler_v2_module, tmp_path: Path
) -> None:
    paths = []
    for i in range(2):
        p = tmp_path / f"loop{i}.mid"
        _make_loop(p, 16)
        paths.append(p)
    out = tmp_path / "state.pkl"
    ckpt = tmp_path / "ckpt"
    res1 = groove_sampler_v2_module.train_streaming(
        paths[:1],
        output=out,
        save_every=1,
        checkpoint_dir=ckpt,
        log_every=1,
    )
    assert res1["kept"] == 1
    res2 = groove_sampler_v2_module.train_streaming(
        paths,
        output=out,
        save_every=1,
        checkpoint_dir=ckpt,
        resume_from=out,
        log_every=1,
    )
    assert res2["kept"] == 2
