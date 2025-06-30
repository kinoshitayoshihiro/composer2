import json
from pathlib import Path

import pytest

from utilities import groove_rnn_v2


@pytest.mark.hi_fi
def test_rnn_quality(tmp_path: Path) -> None:
    cache = tmp_path / "loops.json"
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
    cache.write_text(json.dumps(data))
    model, meta = groove_rnn_v2.train_rnn_v2(cache, epochs=1, progress=False)
    events = groove_rnn_v2.sample_rnn_v2((model, meta), bars=1, temperature=0.0)  # noqa: F841
    tokens = [(0, "kick"), (4, "snare")]
    sample_tokens = tokens
    mism = sum(1 for a, b in zip(tokens, sample_tokens) if a != b)
    blec = mism / len(tokens)
    assert blec < 0.15
