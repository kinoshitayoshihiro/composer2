import subprocess

from utilities.install_utils import run_with_retry


def test_run_with_retry(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_call(cmd):
        calls.append(cmd)
        if len(calls) < 2:
            raise subprocess.CalledProcessError(returncode=1, cmd=cmd)

    monkeypatch.setattr(subprocess, "check_call", fake_call)
    run_with_retry(["echo", "ok"], attempts=3, delay=0.0)
    assert len(calls) == 2
