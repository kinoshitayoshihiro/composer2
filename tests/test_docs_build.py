import subprocess

import pytest

pytestmark = pytest.mark.docs


def test_mkdocs_build(tmp_path):
    subprocess.check_call(["mkdocs", "build", "--strict", "--site-dir", str(tmp_path)])
