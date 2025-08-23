import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_train_phrase_help(tmp_path):
    pytest.importorskip("torch")
    script = Path(__file__).resolve().parents[1] / "scripts" / "train_phrase.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=True,
        capture_output=True,
        env={**os.environ, "ALLOW_LOCAL_IMPORT": "1"},
    )
    assert b"usage" in result.stdout.lower()
