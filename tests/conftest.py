import os;
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import sys
import os
import builtins
import types

# use real dependencies by default
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
if (
    "fastapi" in sys.modules
    and getattr(sys.modules["fastapi"], "__spec__", None) is None
):
    del sys.modules["fastapi"]

# stubs are disabled; tests rely on real packages

import pytest
sys.modules.setdefault("requests", types.ModuleType("requests"))
try:  # pragma: no cover - optional dependency
    from music21 import instrument
except Exception:  # pragma: no cover - fallback stub
    class instrument:  # type: ignore
        Instrument = object

# すべてのテストモジュールで types を利用可能にする
builtins.types = types


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


@pytest.fixture
def rhythm_library():
    from utilities.rhythm_library_loader import load_rhythm_library

    return load_rhythm_library("data/rhythm_library.yml")


def pytest_addoption(parser):
    parser.addoption(
        "--update-golden",
        action="store_true",
        default=False,
        help="update golden midi files",
    )
