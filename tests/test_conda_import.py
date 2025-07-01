import os
import sys
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.packaging


def test_conda_import(tmp_path):
    try:
        subprocess.check_call(["conda-build", "--version"], stdout=subprocess.DEVNULL)
    except Exception:
        pytest.skip("conda-build not available")
    build_dir = tmp_path / "conda"
    subprocess.check_call(["conda-build", "recipe", "--output-folder", str(build_dir)])
    tarball = next(Path(build_dir).rglob("*.tar.bz2"))
    subprocess.check_call([sys.executable, "-m", "pip", "install", tarball])
    __import__("modular_composer")
