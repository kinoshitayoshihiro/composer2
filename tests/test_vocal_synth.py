import sys
import types
from pathlib import Path

import pytest

# Fake sunoai module to avoid real API calls
class FakeClient:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
    def generate(self, *a, **k):
        return b"FAKEWAV"

sys.modules['sunoai'] = types.SimpleNamespace(GenerationClient=FakeClient)

from utilities import vocal_synth


def test_vocal_synth_cli(tmp_path: Path, monkeypatch):
    out = tmp_path / "out.wav"
    result = vocal_synth.main(["--model", "demo", "--output", str(out)])
    assert out.read_bytes() == b"FAKEWAV"
    assert result == out
