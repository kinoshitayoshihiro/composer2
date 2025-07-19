import subprocess
import sys
from pathlib import Path

import pytest

try:
    import torch
except Exception:
    torch = None


@pytest.mark.skipif(torch is None, reason="requires torch")
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


@pytest.mark.skipif(torch is None, reason="requires torch")
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
