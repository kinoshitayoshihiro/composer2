from __future__ import annotations

"""Utility for installing lightweight module stubs used by the tests."""

import sys  # noqa: E402
import types  # noqa: E402
import importlib.util  # noqa: E402
from pathlib import Path  # noqa: E402

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover - optional
    torch = types.ModuleType("torch")
    nn = types.ModuleType("torch.nn")
    utils = types.ModuleType("torch.utils")
    utils.data = types.ModuleType("torch.utils.data")
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
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
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
            sys.modules[name] = mod

    utils_path = Path(__file__).resolve().parent.parent / "utilities" / "stub_utils.py"
    spec = importlib.util.spec_from_file_location("stub_utils", utils_path)
    assert spec and spec.loader
    stub_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stub_utils)  # type: ignore[attr-defined]
    stub_utils.install_stubs(
        force_names=("pytorch_lightning", "music21", "yaml", "pretty_midi")
    )

    pm = sys.modules.get("pretty_midi")
    if pm and not hasattr(pm.PrettyMIDI, "get_tempo_changes"):
        class _PM:
            def __init__(self, *args, **kwargs) -> None:
                pass

            def get_tempo_changes(self, *args, **kwargs):
                return [], []

        pm.PrettyMIDI = _PM
    m21 = sys.modules.get("music21")
    if m21:
        midi = sys.modules.setdefault("music21.midi", types.ModuleType("music21.midi"))
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
            mod.__spec__ = types.SimpleNamespace(loader=None)
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

    if "pytest_asyncio" not in sys.modules:
        import pytest
        import asyncio

        pa = types.ModuleType("pytest_asyncio")

        @pytest.fixture
        def event_loop():
            loop = asyncio.new_event_loop()
            yield loop
            loop.close()

        pa.event_loop = event_loop
        pa.pytest_configure = lambda config: None
        sys.modules["pytest_asyncio"] = pa
