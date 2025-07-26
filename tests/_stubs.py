from __future__ import annotations

"""Utility for installing lightweight module stubs used by the tests."""

import importlib
import importlib.util  # noqa: E402
import sys  # noqa: E402
import types  # noqa: E402
from pathlib import Path  # noqa: E402

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

        def to(self, *_a, **_k):
            return self

    torch.Tensor = _Tensor  # type: ignore[attr-defined]

    class _Module:  # minimal nn.Module replacement
        def __init__(self, *args, **kwargs) -> None:
            pass

        def parameters(self):
            return []

    class _Embedding(_Module):
        pass

    class _GRU(_Module):
        def __call__(self, x, *a, **k):
            return x, None

    class _Linear(_Module):
        pass

    nn.Module = _Module  # type: ignore[attr-defined]
    nn.Embedding = _Embedding  # type: ignore[attr-defined]
    nn.GRU = _GRU  # type: ignore[attr-defined]
    nn.Linear = _Linear  # type: ignore[attr-defined]
    nn.functional = types.SimpleNamespace()  # type: ignore[attr-defined]
    torch.nn = nn  # type: ignore[attr-defined]
    torch.zeros = lambda *a, **k: 0  # type: ignore[attr-defined]
    sys.modules.setdefault("torch", torch)
    sys.modules.setdefault("torch.nn", nn)
    torch.utils = utils  # type: ignore[attr-defined]
    sys.modules.setdefault("torch.utils", utils)
    sys.modules.setdefault("torch.utils.data", utils.data)


def install() -> None:
    """Register lightweight stubs for optional dependencies."""
    for name in ("pretty_midi", "music21", "yaml", "pydantic", "pydantic_settings"):
        spec = importlib.util.find_spec(name)
        if spec is not None:
            continue
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__spec__ = importlib.machinery.ModuleSpec(
                name, loader=importlib.machinery.BuiltinImporter
            )
            if name == "pretty_midi":

                class _PM:
                    def __init__(self, *args, **kwargs) -> None:
                        pass

                    def get_tempo_changes(self, *args, **kwargs):
                        return [], []

                setattr(mod, "PrettyMIDI", _PM)
            if name == "yaml":
                mod.safe_load = lambda *_a, **_k: {}
            if name == "pydantic":

                class _BaseModel:
                    def __init__(self, *a, **k):
                        pass

                mod.BaseModel = _BaseModel
                mod.FilePath = str
            if name == "pydantic_settings":

                class _BaseSettings:
                    pass

                mod.BaseSettings = _BaseSettings
            if name == "music21":
                instr = types.ModuleType("music21.instrument")
                instr.__spec__ = importlib.machinery.ModuleSpec("music21.instrument", loader=importlib.machinery.BuiltinImporter)
                class _Instr:
                    def __init__(self, *a, **k) -> None:
                        pass
                instr.Instrument = _Instr
                mod.instrument = instr
                sys.modules["music21.instrument"] = instr
            sys.modules[name] = mod

    utils_path = Path(__file__).resolve().parent.parent / "utilities" / "stub_utils.py"
    spec = importlib.util.spec_from_file_location("stub_utils", utils_path)
    assert spec and spec.loader
    stub_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stub_utils)  # type: ignore[attr-defined]
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
        if mod is None:
            mod = types.ModuleType(name)
            mod.__spec__ = importlib.machinery.ModuleSpec(
                name, loader=importlib.machinery.BuiltinImporter
            )
            sys.modules[name] = mod
        mod.CRF = _DummyCRF  # type: ignore[attr-defined]
        if name == "pytorch_lightning":

            class _LM(torch.nn.Module):
                def __init__(self) -> None:
                    super().__init__()

                def parameters(self):
                    t = torch.zeros(1)
                    return iter([t])

            mod.LightningDataModule = object  # type: ignore[attr-defined]
            mod.LightningModule = _LM  # type: ignore[attr-defined]
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

    # Additional lightweight stubs used in tests. Avoid stubbing FastAPI so
    # that its absence is detected properly.
    for name in (
        "uvicorn",
        "websockets",
        "streamlit",
        "mido",
        "tomli",
        "transformers",
    ):
        if name in sys.modules or importlib.util.find_spec(name) is not None:
            continue
        mod = types.ModuleType(name)
        mod.__spec__ = importlib.machinery.ModuleSpec(
            name, loader=importlib.machinery.BuiltinImporter, is_package=True
        )
        sys.modules[name] = mod
        if name == "tomli":
            mod.loads = lambda s, *a, **k: {}
        if name == "transformers":
            class _Model:
                pass

            mod.Wav2Vec2Model = _Model

    if "transformers" not in sys.modules and importlib.util.find_spec("transformers") is None:
        mod = types.ModuleType("transformers")
        mod.__spec__ = importlib.machinery.ModuleSpec(
            "transformers", loader=importlib.machinery.BuiltinImporter, is_package=True
        )
        sys.modules["transformers"] = mod
        class _Model:
            pass

        mod.Wav2Vec2Model = _Model
