import json
from pathlib import Path
import pytest

try:
    import torch  # type: ignore
except ImportError:  # pragma: no cover - optional
    torch = None  # type: ignore

pytest.importorskip("pytorch_lightning")
pytestmark = pytest.mark.skipif(torch is None, reason="torch not installed")

from utilities import groove_sampler_rnn


def _make_loop(path: Path) -> None:
    data = {
        "ppq": 480,
        "resolution": 16,
        "data": [
            {
                "file": "a.mid",
                "tokens": [(i, "kick", 100, 0) for i in range(16)],
                "tempo_bpm": 120.0,
                "bar_beats": 4,
                "section": "verse",
                "heat_bin": 0,
                "intensity": "mid",
            }
        ],
    }
    path.write_text(json.dumps(data))


def test_offsets(tmp_path: Path) -> None:
    cache = tmp_path / "loops.json"
    _make_loop(cache)
    model, meta = groove_sampler_rnn.train(cache, epochs=1)
    events = groove_sampler_rnn.sample(model, meta, bars=2, temperature=0.0)
    off_max = max(ev["offset"] for ev in events)
    assert off_max == pytest.approx(7.5, rel=0.5)
