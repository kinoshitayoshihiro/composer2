import json
import logging
import importlib.util
import sys
import types
from pathlib import Path

import pytest

# Optional heavy deps
pytest.importorskip("numpy")
pytest.importorskip("pretty_midi")
pytest.importorskip("mido")

ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


@pytest.fixture
def groove_sampler_v2_aux_modules(
    stub_utilities, monkeypatch: pytest.MonkeyPatch
):
    aux_mod = _load_module("tests.aux_vocab", ROOT / "utilities" / "aux_vocab.py")
    gm_mod = _load_module("tests.gm_perc_map", ROOT / "utilities" / "gm_perc_map.py")
    stubs = {
        "loop_ingest": types.SimpleNamespace(load_meta=lambda *a, **k: {}),
        "groove_sampler": types.SimpleNamespace(
            _PITCH_TO_LABEL={},
            _iter_drum_notes=lambda *a, **k: [],
            infer_resolution=lambda *a, **k: 480,
        ),
        "aux_vocab": aux_mod,
        "conditioning": types.SimpleNamespace(
            apply_feel_bias=lambda *a, **k: None,
            apply_kick_pattern_bias=lambda *a, **k: None,
            apply_style_bias=lambda *a, **k: None,
            apply_velocity_bias=lambda *a, **k: None,
        ),
        "hash_utils": types.SimpleNamespace(hash_ctx=lambda *a, **k: 0),
        "gm_perc_map": gm_mod,
        "ngram_store": types.SimpleNamespace(
            BaseNGramStore=object,
            MemoryNGramStore=object,
            SQLiteNGramStore=object,
        ),
    }
    if "pkg_resources" not in sys.modules:
        monkeypatch.setitem(
            sys.modules,
            "pkg_resources",
            types.SimpleNamespace(resource_stream=lambda *a, **k: None),
        )
    module_path = ROOT / "utilities" / "groove_sampler_v2.py"
    with stub_utilities(
        "utilities.groove_sampler_v2",
        module_path,
        package_attrs={"__path__": []},
        submodules=stubs,
    ) as module:
        yield aux_mod, module


def _make_model(aux_mod, gs_mod, tmp_path: Path) -> Path:
    aux = aux_mod.AuxVocab()
    aux.encode({"mood": "happy"})
    model = gs_mod.NGramModel(
        n=1,
        resolution=4,
        resolution_coarse=4,
        state_to_idx={},
        idx_to_state=[],
        freq=[],
        bucket_freq={},
        ctx_maps=[],
        prob_paths=None,
        prob=None,
        aux_vocab=aux,
        version=2,
        file_weights=None,
        files_scanned=0,
        files_skipped=0,
        total_events=0,
        hash_buckets=16,
    )
    path = tmp_path / "model.pkl"
    gs_mod.save(model, path)
    return path


def test_load_uses_embedded_vocab(
    groove_sampler_v2_aux_modules, tmp_path: Path
) -> None:
    aux_mod, gs_mod = groove_sampler_v2_aux_modules
    model_path = _make_model(aux_mod, gs_mod, tmp_path)
    model = gs_mod.load(model_path)
    assert model.aux_vocab.id_to_str[-1] == "mood=happy"


def test_load_override_vocab(
    groove_sampler_v2_aux_modules, tmp_path: Path
) -> None:
    aux_mod, gs_mod = groove_sampler_v2_aux_modules
    model_path = _make_model(aux_mod, gs_mod, tmp_path)
    aux_path = tmp_path / "aux.json"
    aux_path.write_text(json.dumps(["", "<UNK>", "mood=sad"]))
    model = gs_mod.load(model_path, aux_vocab_path=aux_path)
    assert model.aux_vocab.id_to_str[-1] == "mood=sad"


def test_load_bad_aux_falls_back(
    groove_sampler_v2_aux_modules, tmp_path: Path, caplog
) -> None:
    aux_mod, gs_mod = groove_sampler_v2_aux_modules
    model_path = _make_model(aux_mod, gs_mod, tmp_path)
    bad_path = tmp_path / "missing.json"
    with caplog.at_level(logging.WARNING):
        model = gs_mod.load(model_path, aux_vocab_path=bad_path)
    assert model.aux_vocab.id_to_str[-1] == "mood=happy"
    assert any("failed to load aux vocab" in r.message for r in caplog.records)
