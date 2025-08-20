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

# Load modules directly to avoid heavy package imports
ROOT = Path(__file__).resolve().parents[1]

aux_spec = importlib.util.spec_from_file_location(
    "utilities.aux_vocab", ROOT / "utilities" / "aux_vocab.py"
)
aux_mod = importlib.util.module_from_spec(aux_spec)
sys.modules["utilities.aux_vocab"] = aux_mod
aux_spec.loader.exec_module(aux_mod)

gm_spec = importlib.util.spec_from_file_location(
    "utilities.gm_perc_map", ROOT / "utilities" / "gm_perc_map.py"
)
gm_mod = importlib.util.module_from_spec(gm_spec)
sys.modules["utilities.gm_perc_map"] = gm_mod
gm_spec.loader.exec_module(gm_mod)

gs_spec = importlib.util.spec_from_file_location(
    "utilities.groove_sampler_v2", ROOT / "utilities" / "groove_sampler_v2.py"
)
gs_mod = importlib.util.module_from_spec(gs_spec)
# Minimal package stubs required by groove_sampler_v2
pkg = types.ModuleType("utilities")
pkg.__path__ = []  # type: ignore[attr-defined]
pkg.loop_ingest = types.SimpleNamespace(load_meta=lambda *a, **k: {})
pkg.groove_sampler = types.SimpleNamespace(
    _PITCH_TO_LABEL={}, _iter_drum_notes=lambda *a, **k: [], infer_resolution=lambda *a, **k: 480
)
pkg.aux_vocab = aux_mod
pkg.conditioning = types.SimpleNamespace(
    apply_feel_bias=lambda *a, **k: None,
    apply_kick_pattern_bias=lambda *a, **k: None,
    apply_style_bias=lambda *a, **k: None,
    apply_velocity_bias=lambda *a, **k: None,
)
pkg.hash_utils = types.SimpleNamespace(hash_ctx=lambda *a, **k: 0)
pkg.gm_perc_map = gm_mod
pkg.ngram_store = types.SimpleNamespace(
    BaseNGramStore=object, MemoryNGramStore=object, SQLiteNGramStore=object
)
sys.modules["utilities"] = pkg
sys.modules["utilities.loop_ingest"] = pkg.loop_ingest
sys.modules["utilities.groove_sampler"] = pkg.groove_sampler
sys.modules["utilities.aux_vocab"] = aux_mod
sys.modules["utilities.conditioning"] = pkg.conditioning
sys.modules["utilities.hash_utils"] = pkg.hash_utils
sys.modules["utilities.gm_perc_map"] = gm_mod
sys.modules["utilities.ngram_store"] = pkg.ngram_store
sys.modules["utilities.groove_sampler_v2"] = gs_mod
# stub optional dependency used by pretty_midi
sys.modules.setdefault("pkg_resources", types.SimpleNamespace(resource_stream=lambda *a, **k: None))

gs_spec.loader.exec_module(gs_mod)

AuxVocab = aux_mod.AuxVocab
NGramModel = gs_mod.NGramModel
save = gs_mod.save
load = gs_mod.load


def _make_model(tmp_path: Path) -> Path:
    aux = AuxVocab()
    aux.encode({"mood": "happy"})
    model = NGramModel(
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
    save(model, path)
    return path


def test_load_uses_embedded_vocab(tmp_path: Path) -> None:
    model_path = _make_model(tmp_path)
    model = load(model_path)
    assert model.aux_vocab.id_to_str[-1] == "mood=happy"


def test_load_override_vocab(tmp_path: Path) -> None:
    model_path = _make_model(tmp_path)
    aux_path = tmp_path / "aux.json"
    aux_path.write_text(json.dumps(["", "<UNK>", "mood=sad"]))
    model = load(model_path, aux_vocab_path=aux_path)
    assert model.aux_vocab.id_to_str[-1] == "mood=sad"


def test_load_bad_aux_falls_back(tmp_path: Path, caplog) -> None:
    model_path = _make_model(tmp_path)
    bad_path = tmp_path / "missing.json"
    with caplog.at_level(logging.WARNING):
        model = load(model_path, aux_vocab_path=bad_path)
    assert model.aux_vocab.id_to_str[-1] == "mood=happy"
    assert any("failed to load aux vocab" in r.message for r in caplog.records)
