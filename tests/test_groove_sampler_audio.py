import numpy as np
import pytest
import soundfile as sf
from pathlib import Path

from utilities import groove_sampler_v2

pytestmark = pytest.mark.requires_audio


def _make_kick_wav(path: Path) -> None:
    sr = 16000
    y = np.zeros(int(sr * 2), dtype=np.float32)
    for i in range(4):
        y[int(i * 0.5 * sr)] = 1.0
    sf.write(path, y, sr)


def test_train_with_audio(tmp_path: Path) -> None:
    wav = tmp_path / "kick.wav"
    _make_kick_wav(wav)
    model = groove_sampler_v2.train(tmp_path, n=2)
    assert len(model.idx_to_state) > 0
