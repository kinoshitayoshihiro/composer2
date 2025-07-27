from __future__ import annotations

"""Utility for installing lightweight module stubs used by the tests."""

import importlib
import importlib.util  # noqa: E402
import sys  # noqa: E402
import types  # noqa: E402
from pathlib import Path  # noqa: E402
import os

try:
    import torch  # type: ignore
    if isinstance(getattr(torch, "__spec__", None), importlib.machinery.ModuleSpec) and isinstance(
        torch.__spec__.loader, importlib.machinery.BuiltinImporter
    ):
        raise ImportError("stub torch")
except Exception:  # pragma: no cover - optional
    torch = types.ModuleType("torch")
    torch.__spec__ = importlib.machinery.ModuleSpec(
        "torch", loader=importlib.machinery.BuiltinImporter, is_package=True
    )
    nn = types.ModuleType("torch.nn")
    nn.__spec__ = importlib.machinery.ModuleSpec(
        "torch.nn", loader=importlib.machinery.BuiltinImporter, is_package=True
    )
    utils = types.ModuleType("torch.utils")
    utils.__spec__ = importlib.machinery.ModuleSpec(
        "torch.utils", loader=importlib.machinery.BuiltinImporter, is_package=True
    )
    utils.data = types.ModuleType("torch.utils.data")
    utils.data.__spec__ = importlib.machinery.ModuleSpec(
        "torch.utils.data", loader=importlib.machinery.BuiltinImporter, is_package=True
    )

    class _DataLoader:
        def __init__(self, dataset, *args, **kwargs) -> None:
            self.dataset = dataset

        def __iter__(self):
            return iter(self.dataset)

    utils.data.DataLoader = _DataLoader  # type: ignore[attr-defined]

    class _Dataset:
        def __class_getitem__(cls, _item):
            return cls

    utils.data.Dataset = _Dataset  # type: ignore[attr-defined]

    class _WeightedRandomSampler:
        def __init__(self, *a, **k):
            pass

        def __iter__(self):
            return iter([])

    utils.data.WeightedRandomSampler = _WeightedRandomSampler  # type: ignore[attr-defined]

    class _Subset(_Dataset):
        def __init__(self, dataset, indices) -> None:
            self.dataset = dataset
            self.indices = list(indices)

        def __getitem__(self, idx):
            return self.dataset[self.indices[idx]]

        def __len__(self) -> int:  # pragma: no cover - simple len
            return len(self.indices)

    utils.data.Subset = _Subset  # type: ignore[attr-defined]

    class _Tensor:
        def __init__(self, *shape) -> None:
            self.shape = shape

        def zero_(self):
            return self

        @property
        def data(self):
            return self

        def __getitem__(self, _):
            return _Tensor(1)

        def __setitem__(self, _, __):
            pass

        def __len__(self):
            return self.shape[0] if self.shape else 0

        def dim(self):
            return len(self.shape)

        def new_zeros(self, shape):
            return _Tensor(*shape)

        def to(self, *_a, **_k):
            return self

        def unsqueeze(self, dim):
            if dim == 0:
                return _Tensor(1, *self.shape)
            return _Tensor(*self.shape)

        def transpose(self, dim0, dim1):
            return self

        def argmax(self, dim=None):
            return _Tensor(1)

        def __iter__(self):
            return iter(range(len(self)))

        def __int__(self):
            return 0

    torch.Tensor = _Tensor  # type: ignore[attr-defined]

    class _Module:  # minimal nn.Module replacement
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __call__(self, *a, **k):
            if hasattr(self, "forward"):
                return self.forward(*a, **k)
            return None

        def parameters(self):
            return []

    class _Embedding(_Module):
        def __init__(self, num_embeddings, embedding_dim, *a, **k):
            self.embedding_dim = embedding_dim
        def forward(self, idx):
            return _Tensor(len(idx), self.embedding_dim)

    class _GRU(_Module):
        def __call__(self, x, *a, **k):
            return x, None

    class _LSTM(_Module):
        def forward(self, x, *a, **k):
            return x, None

    class _Linear(_Module):
        def __init__(self, *a, **k) -> None:
            self.weight = _Tensor()
            self.bias = _Tensor()
        def forward(self, x):
            return x

    class _LayerNorm(_Module):
        def forward(self, x):
            return x

    class _Dropout(_Module):
        def __init__(self, *a, **k) -> None:
            pass

        def forward(self, x):
            return x

    nn.Module = _Module  # type: ignore[attr-defined]
    nn.Embedding = _Embedding  # type: ignore[attr-defined]
    nn.GRU = _GRU  # type: ignore[attr-defined]
    nn.Linear = _Linear  # type: ignore[attr-defined]
    nn.LayerNorm = _LayerNorm  # type: ignore[attr-defined]
    nn.Dropout = _Dropout  # type: ignore[attr-defined]
    nn.LSTM = _LSTM  # type: ignore[attr-defined]
    nn.functional = types.SimpleNamespace(
        pad=lambda x, pad: x,
        log_softmax=lambda x, dim=None: x,
    )  # type: ignore[attr-defined]
    def _cat(seq, dim=0):  # pragma: no cover - very small fake concat
        if not seq:
            return _Tensor()
        if len(seq) == 1:
            return seq[0]
        base = getattr(seq[0], "shape", ())
        if not base:
            return seq[0]
        total = sum(getattr(t, "shape", base)[dim] for t in seq)
        new_shape = list(base)
        new_shape[dim] = total
        return _Tensor(*new_shape)

    torch.cat = _cat  # type: ignore[attr-defined]
    torch.nn = nn  # type: ignore[attr-defined]
    torch.zeros = lambda *a, **k: _Tensor(*a)  # type: ignore[attr-defined]
    torch.long = "long"  # type: ignore[attr-defined]
    torch.float32 = "float32"  # type: ignore[attr-defined]
    torch.tensor = lambda data, *a, **k: _Tensor(len(data))  # type: ignore[attr-defined]
    class _NoGrad:
        def __enter__(self):
            pass
        def __exit__(self, exc_type, exc, tb):
            return False

    torch.no_grad = lambda: _NoGrad()  # type: ignore[attr-defined]
    sys.modules.setdefault("torch", torch)
    sys.modules.setdefault("torch.nn", nn)
    torch.utils = utils  # type: ignore[attr-defined]
    sys.modules.setdefault("torch.utils", utils)
    sys.modules.setdefault("torch.utils.data", utils.data)
    # -------------------- torch.distributed (no-op) --------------------
    if "torch.distributed" not in sys.modules:
        try:
            import torch.distributed as _dist  # type: ignore
        except Exception:
            dist = types.ModuleType("torch.distributed")
            dist.__spec__ = importlib.machinery.ModuleSpec(
                "torch.distributed", loader=importlib.machinery.BuiltinImporter, is_package=True
            )
            def _noop(*_a, **_k):
                pass
            dist.is_available = lambda: False
            dist.init_process_group = _noop
            dist.destroy_process_group = _noop
            dist.barrier = _noop
            dist.get_rank = lambda: 0
            dist.get_world_size = lambda: 1
            class _PG:
                pass
            dist.ProcessGroup = _PG
            sys.modules.setdefault("torch.distributed", dist)
            torch.distributed = dist  # type: ignore[attr-defined]

    # -------------------- torch._dynamo (no-op) --------------------
    dyn = types.ModuleType("torch._dynamo")
    dyn.__spec__ = importlib.machinery.ModuleSpec(
        "torch._dynamo", loader=importlib.machinery.BuiltinImporter, is_package=True
    )
    dyn.optimize = lambda *a, **k: (lambda f: f)
    dyn.allow_in_graph = lambda *_a, **_k: False
    sys.modules.setdefault("torch._dynamo", dyn)

    # -------------------- torch.library / torch._library (no-op) --------------------
    lib = types.ModuleType("torch.library")
    lib.__spec__ = importlib.machinery.ModuleSpec(
        "torch.library", loader=importlib.machinery.BuiltinImporter, is_package=True
    )
    def _noop_register(*_a, **_k):
        return None
    class _FakeLibrary:
        def __init__(self, *a, **k):
            pass
        def define(self, *a, **k):
            pass
        _register_fake = _noop_register
    lib.register = _noop_register
    lib.Library = _FakeLibrary
    sys.modules.setdefault("torch.library", lib)

    _lib_pkg = types.ModuleType("torch._library")
    _lib_pkg.__spec__ = importlib.machinery.ModuleSpec(
        "torch._library", loader=importlib.machinery.BuiltinImporter, is_package=True
    )
    utils_mod = types.ModuleType("torch._library.utils")
    utils_mod.__spec__ = importlib.machinery.ModuleSpec(
        "torch._library.utils", loader=importlib.machinery.BuiltinImporter, is_package=False
    )
    utils_mod.get_source = lambda *a, **k: ""
    _lib_pkg.utils = utils_mod
    sys.modules.setdefault("torch._library", _lib_pkg)
    sys.modules.setdefault("torch._library.utils", utils_mod)


def install() -> None:
    """Register lightweight stubs for optional dependencies."""
    if "fastapi" in sys.modules and getattr(sys.modules["fastapi"], "__spec__", None) is None:
        spec = importlib.util.find_spec("fastapi")
        if spec is not None:
            sys.modules["fastapi"].__spec__ = spec
            sys.modules["fastapi"].__file__ = getattr(spec, "origin", "<fastapi>")

    utils_path = Path(__file__).resolve().parent.parent / "utilities" / "stub_utils.py"
    spec = importlib.util.spec_from_file_location("stub_utils", utils_path)
    assert spec and spec.loader
    stub_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stub_utils)  # type: ignore[attr-defined]
    _dummy_class = stub_utils._dummy_class  # type: ignore[attr-defined]
    _dummy_module = stub_utils._dummy_module  # type: ignore[attr-defined]

    sig = sys.modules.get("scipy.signal")
    if sig and getattr(sig, "__spec__", None) is None:
        sig.__spec__ = importlib.machinery.ModuleSpec(
            "scipy.signal", loader=importlib.machinery.BuiltinImporter, is_package=True
        )
        sig.__spec__.submodule_search_locations = []
        sig.__package__ = "scipy.signal"
        sig.__path__ = []

    stub_utils.install_stubs()

    pm = sys.modules.get("pretty_midi")
    if pm and not hasattr(pm.PrettyMIDI, "get_tempo_changes"):
        import numpy as np

        class _PM:
            def __init__(self, *args, **kwargs) -> None:
                pass

            def get_tempo_changes(self, *args, **kwargs):
                return np.asarray([]), np.asarray([])

        pm.PrettyMIDI = _PM
    m21 = sys.modules.get("music21")
    if m21:
        sys.modules.setdefault("music21.midi", types.ModuleType("music21.midi"))
        translate = sys.modules.setdefault(
            "music21.midi.translate", types.ModuleType("music21.midi.translate")
        )
        tie_mod = sys.modules.setdefault("music21.tie", types.ModuleType("music21.tie"))
        tie_mod.__spec__ = importlib.machinery.ModuleSpec(
            "music21.tie", loader=importlib.machinery.BuiltinImporter, is_package=False
        )
        class _Tie(_dummy_class):
            @classmethod
            def __mro_entries__(cls, bases):
                return (cls,)
        tie_mod.Tie = _Tie

        def _m21_to_pm(score):
            return pm.PrettyMIDI() if pm else None

        translate.m21ObjectToPrettyMIDI = _m21_to_pm

    class _DummyCRF:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __call__(self, *args, **kwargs) -> float:
            return 0.0

        def decode(self, emissions, mask=None):
            return [[0] * len(emissions[0])]

        viterbi_decode = decode

    for name in (
        "pkg_resources",
        "scipy",
        "scipy.signal",
        "torchcrf",
        "torch_crf",
        "TorchCRF",
        "pytorch_lightning",
    ):
        mod = sys.modules.get(name)
        if mod is None or getattr(mod, "__spec__", None) is None or not importlib.util.find_spec(name):
            mod = types.ModuleType(name)
            mod.__file__ = f"<stub:{name}>"
            is_pkg = name in {"scipy", "scipy.signal"}
            mod.__spec__ = importlib.machinery.ModuleSpec(
                name,
                loader=importlib.machinery.BuiltinImporter,
                is_package=is_pkg,
            )
            if is_pkg:
                mod.__spec__.submodule_search_locations = []
                mod.__package__ = name
                mod.__path__ = []
            sys.modules[name] = mod
        mod.CRF = _DummyCRF  # type: ignore[attr-defined]
        if name == "pytorch_lightning":

            class _LM(torch.nn.Module):
                def __init__(self) -> None:
                    super().__init__()

                def parameters(self):
                    t = torch.zeros(1)
                    return iter([t])

            class _Trainer:
                def __init__(self, *a, **k) -> None:
                    pass

                def fit(self, *a, **k):
                    return None

                def validate(self, *a, **k):
                    return None

                def test(self, *a, **k):
                    return None

            mod.LightningDataModule = object  # type: ignore[attr-defined]
            mod.LightningModule = _LM  # type: ignore[attr-defined]
            mod.Trainer = _Trainer  # type: ignore[attr-defined]
        sys.modules.setdefault(name, mod)
        if name == "scipy.signal":
            mod.resample_poly = lambda x, up, down, axis=0: x

    # Ensure basic signal helpers if scipy is missing
    if "scipy" not in sys.modules:
        sig = types.ModuleType("scipy.signal")
        def _dummy_filter(*_a, **_k):
            return []
        sig.butter = _dummy_filter
        sig.lfilter = _dummy_filter
        sig.filtfilt = _dummy_filter
        sig.find_peaks = lambda *_a, **_k: ([], {})
        sig.resample_poly = lambda x, up, down, axis=0: x
        sys.modules["scipy"] = types.ModuleType("scipy")
        sys.modules["scipy.signal"] = sig
    else:
        try:
            from scipy.signal import butter as _bf, lfilter as _lf, filtfilt as _ff, find_peaks as _fp
        except Exception:
            sig = sys.modules.setdefault("scipy.signal", types.ModuleType("scipy.signal"))
            def _dummy_filter(*_a, **_k):
                return []
            sig.butter = _dummy_filter
            sig.lfilter = _dummy_filter
            sig.filtfilt = _dummy_filter
            sig.find_peaks = lambda *_a, **_k: ([], {})
            sig.resample_poly = lambda x, up, down, axis=0: x

    if "pytest_asyncio" not in sys.modules:
        import asyncio

        import pytest

        pa = types.ModuleType("pytest_asyncio")

        @pytest.fixture
        def event_loop():
            loop = asyncio.new_event_loop()
            yield loop
            loop.close()

        pa.event_loop = event_loop

        def pytest_addoption(parser):
            parser.addoption(
                "--asyncio-mode",
                action="store",
                default="auto",
                help="dummy asyncio mode option for stub",
            )

        pa.pytest_addoption = pytest_addoption
        pa.pytest_configure = lambda config: None
        sys.modules["pytest_asyncio"] = pa

    # ------------------------------------------------------------------
    # 強制スタブ: transformers / torchmetrics 系は常にスタブ化して
    # '_dummy_module' 由来の TypeError を防止する
    # ------------------------------------------------------------------
    heavy = (
        "uvicorn",
        "websockets",
        "streamlit",
        "mido",
        "tomli",
        "transformers",
        "pytorch_lightning",
        "lightning_fabric",
        "lightning_utilities",
        "tensorboard",
        "tensorboardX",
        "torch.distributed",
        "torch._dynamo",
        "torch.library",
        "torch._library",
        "torchmetrics",
        "torchmetrics.functional",
        "torchmetrics.functional.text",
        "librosa",
    )

    for name in heavy:
        force = (
            name.startswith("transformers")
            or name.startswith("torchmetrics")
            or name in {"streamlit", "uvicorn", "websockets", "mido",
                        "pytorch_lightning", "lightning_fabric", "lightning_utilities",
                        "tensorboard", "tensorboardX", "torch._dynamo"}
        )
        if name.startswith("transformers") and os.getenv("COMPOSER_USE_DUMMY_TRANSFORMERS", "0") != "1":
            force = False

        # transformers/torchmetrics は必ず潰す。それ以外は存在確認して無ければ stub
        if not force:
            if name in sys.modules and getattr(sys.modules[name], "__spec__", None):
                continue
            if importlib.util.find_spec(name):
                continue

        # 既にロード済みなら上書きしない
        if name in sys.modules:
            mod_in_sys = sys.modules[name]
            mod_file = getattr(mod_in_sys, "__file__", "")
            if not isinstance(mod_file, str):
                mod_file = ""
            if force and mod_file.startswith("<stub:"):
                continue
            if not force and getattr(mod_in_sys, "__spec__", None):
                continue
            del sys.modules[name]

        mod = types.ModuleType(name)
        mod.__file__ = f"<stub:{name}>"
        mod.__spec__ = importlib.machinery.ModuleSpec(
            name, loader=importlib.machinery.BuiltinImporter, is_package=True
        )
        mod.__spec__.submodule_search_locations = []
        mod.__package__ = name
        mod.__path__ = []
        sys.modules[name] = mod

        if name == "tomli":
            mod.loads = lambda s, *a, **k: {}

        if name.startswith("transformers"):
            class _Dummy:
                def __getattr__(self, _):
                    return self

                def __call__(self, *a, **k):
                    return self

                @classmethod
                def from_pretrained(cls, *a, **k):
                    return cls()

            mod.Wav2Vec2Model = _Dummy
            mod.AutoTokenizer = _Dummy
            mod.AutoModel = _Dummy
            mod.__getattr__ = lambda _n: _Dummy

        if name.startswith("torchmetrics"):
            class _TM:
                def __getattr__(self, _):
                    return self

                def __call__(self, *a, **k):
                    return 0

            mod.__getattr__ = lambda _n: _TM()

        # lightning_utilities/tensorboard 簡易スタブ
        if name == "lightning_utilities":
            mod.rank_zero = types.SimpleNamespace(  # type: ignore
                rank_zero_only=lambda f: f,
                rank_zero_warn=lambda *a, **k: None,
                rank_zero_info=lambda *a, **k: None,
            )
        if name.startswith("tensorboard"):
            mod.SummaryWriter = _dummy_class

        if name == "librosa":
            mod.load = lambda *_a, **_k: ([], 22050)

