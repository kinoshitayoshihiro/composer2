import json
import random
from pathlib import Path
import pytest

pytest.importorskip("torch")

from utilities import groove_sampler_rnn


def _make_loop(path: Path) -> None:
    data = {
        "ppq": 480,
        "resolution": 16,
        "data": [
            {
                "file": "a.mid",
                "tokens": [(0, "kick", 100, 0), (4, "snare", 100, 0)],
                "tempo_bpm": 120.0,
                "bar_beats": 4,
                "section": "verse",
                "heat_bin": 0,
                "intensity": "mid",
            }
        ],
    }
    path.write_text(json.dumps(data))


def test_rnn_smoke(tmp_path: Path) -> None:
    cache = tmp_path / "loops.json"
    _make_loop(cache)
    model, meta = groove_sampler_rnn.train(cache, epochs=1)
    out = groove_sampler_rnn.sample(model, meta, bars=1)
    assert len(out) >= 8
