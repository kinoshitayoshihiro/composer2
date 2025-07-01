import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.packaging


def test_wheel_import(tmp_path):
    wheel_dir = tmp_path / "wheel"
    subprocess.check_call([sys.executable, "-m", "build", "--wheel", "--outdir", str(wheel_dir)])
    wheel = next(wheel_dir.glob("*.whl"))
    subprocess.check_call([sys.executable, "-m", "pip", "install", str(wheel)])
    __import__("modular_composer")
