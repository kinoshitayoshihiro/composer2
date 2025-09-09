import subprocess
import sys
import pytest

def test_auto_tag_help() -> None:
    pytest.importorskip("pretty_midi")
    pytest.importorskip("yaml")
    result = subprocess.run(
        [sys.executable, "-m", "tools.auto_tag_loops", "--help"],
        check=True,
        capture_output=True,
    )
    assert b"usage" in result.stdout.lower()
