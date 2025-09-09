import json
from pathlib import Path
import pytest

from utilities import groove_sampler_rnn

try:
    import torch  # type: ignore
except ImportError:  # pragma: no cover - optional
    torch = None  # type: ignore

pytest.importorskip("pytorch_lightning")
pytestmark = pytest.mark.skipif(torch is None, reason="torch not installed")


def _make_loop(path: Path) -> None:
    tokens = []
    for i in range(64):
        lbl = "kick" if i % 2 == 0 else "snare"
        tokens.append((i % 16, lbl, 100, 0))



@pytest.mark.hi_fi
def test_rnn_quality(tmp_path: Path) -> None:
    data = {
        "ppq": 480,
        "resolution": 16,
        "data": [
            {
                "file": "a.mid",
                "tokens": tokens,
                "tempo_bpm": 120.0,
                "bar_beats": 4,
                "section": "verse",
                "heat_bin": 0,
                "intensity": "mid",
            }
        ],
    }
    path.write_text(json.dumps(data))


def test_rnn_quality(tmp_path: Path) -> None:
    cache = tmp_path / "loops.json"
    _make_loop(cache)
    model, meta = groove_sampler_rnn.train(cache, epochs=3)
    events = groove_sampler_rnn.sample(model, meta, bars=4, temperature=0.0)
    expected = ["kick" if i % 2 == 0 else "snare" for i in range(64)]
    actual = [ev["instrument"] for ev in events[:64]]
    mism = sum(e != a for e, a in zip(expected, actual)) / 64
    assert mism < 0.6
