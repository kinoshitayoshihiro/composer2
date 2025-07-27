import sys
import types
import importlib.util
import importlib.machinery
from importlib.util import spec_from_loader
import os

# Heavy import を抑止（transformers が torch.distributed.* を引っ張るのを防ぐ）
os.environ.setdefault("COMPOSER_USE_DUMMY_TRANSFORMERS", "1")
os.environ.setdefault("TRANSFORMERS_NO_TORCH_FSDP", "1")
os.environ.setdefault("TRANSFORMERS_NO_TORCH_DYNAMO", "1")
os.environ.setdefault("PYTORCH_DISABLE_DYNAMO", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# 既に読み込まれた heavy モジュールを掃除
for _m in (
    "pytorch_lightning",
    "lightning_fabric",
    "lightning_utilities",
    "tensorboard",
    "tensorboardX",
    "torch._dynamo",
    "torch.distributed.fsdp",
    "torch.distributed.tensor",
    "torch.library",
    "torch._library",
):
    sys.modules.pop(_m, None)

# --- kill broken fastapi stub ASAP (before anything else imports it) ---
if "fastapi" in sys.modules and getattr(sys.modules["fastapi"], "__spec__", None) is None:
    del sys.modules["fastapi"]

# stub を必ずインストール（他のモジュールより前に）
from ._stubs import install as _install_stubs  # noqa: E402

_install_stubs()

import pytest
from music21 import instrument


@pytest.fixture(autouse=True)
def stub_optional_deps():
    """Install lightweight stubs for truly optional packages."""
    _install_stubs()
    if importlib.util.find_spec("fastapi") is None:
        sys.modules.pop("fastapi", None)

    def _stub_pkg(name: str, attrs: dict | None = None) -> None:
        # leave real packages untouched
        try:
            if importlib.util.find_spec(name):
                return
        except ValueError:
            pass
        if name in sys.modules and getattr(sys.modules[name], "__spec__", None):
            return

        mod = sys.modules.get(name, types.ModuleType(name))
        spec = spec_from_loader(name, loader=None, is_package=True)
        spec.submodule_search_locations = []
        mod.__spec__ = spec
        mod.__package__ = name
        mod.__path__ = []
        if attrs:
            for k, v in attrs.items():
                setattr(mod, k, v)
        sys.modules[name] = mod

    for pkg in ("uvicorn", "websockets", "streamlit"):
        _stub_pkg(pkg)

    st = sys.modules.get("streamlit")
    if st and not hasattr(st, "cache_data"):
        st.cache_data = lambda func: func


@pytest.fixture
def _basic_gen():
    """Basic GuitarGenerator fixture for testing."""
    from generator.guitar_generator import GuitarGenerator

    def _create_generator(**kwargs):
        # デフォルト設定
        default_args = {
            "global_settings": {},
            "default_instrument": instrument.Guitar(),
            "part_name": "g",
            "global_tempo": 120,
            "global_time_signature": "4/4",
            "global_key_signature_tonic": "C",
            "global_key_signature_mode": "major",
        }
        # kwargsでデフォルト設定を上書き
        default_args.update(kwargs)

        return GuitarGenerator(**default_args)

    return _create_generator

