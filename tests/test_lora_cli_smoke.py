import subprocess
import sys
from pathlib import Path

import pytest

try:
    import torch
except Exception:
    torch = None

try:
    import transformers  # type: ignore[unused-import]
except Exception:
    transformers = None

try:
    import peft  # type: ignore[unused-import]
except Exception:
    peft = None

requires_lora_stack = pytest.mark.skipif(
    torch is None or transformers is None or peft is None,
    reason="requires torch, transformers and peft",
)


@requires_lora_stack
def test_piano_lora_smoke(tmp_path: Path):
    dummy = tmp_path / "dummy.jsonl"
    dummy.write_text('{"tokens": [0,0,0]}\n' * 4)
    out = tmp_path / "model"
    subprocess.check_call(
        [
            sys.executable,
            "train_piano_lora.py",
            "--data",
            str(dummy),
            "--out",
            str(out),
            "--steps",
            "2",
            "--safe",
        ]
    )
    assert (out / "adapter_model.safetensors").exists() or (
        out / "adapter_model.bin"
    ).exists()


@requires_lora_stack
def test_sax_lora_smoke(tmp_path: Path):
    dummy = tmp_path / "dummy.jsonl"
    dummy.write_text('{"tokens": [0,0,0]}\n' * 4)
    out = tmp_path / "model"
    subprocess.check_call(
        [
            sys.executable,
            "train_sax_lora.py",
            "--data",
            str(dummy),
            "--out",
            str(out),
            "--steps",
            "2",
            "--safe",
        ]
    )
    assert (out / "adapter_model.safetensors").exists() or (
        out / "adapter_model.bin"
    ).exists()
