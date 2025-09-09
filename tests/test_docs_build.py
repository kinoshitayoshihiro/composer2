import subprocess
import shutil
import pytest

pytestmark = pytest.mark.docs


def test_mkdocs_build(tmp_path):
    if shutil.which("mkdocs") is None:
        pytest.skip("mkdocs not installed")
    subprocess.check_call(["mkdocs", "build", "--strict", "--site-dir", str(tmp_path)])
