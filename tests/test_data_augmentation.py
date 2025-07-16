import numpy as np
import soundfile as sf
from pathlib import Path

from utilities import data_augmentation


def test_augment_wav_dir(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    sr = 16000
    y = np.zeros(sr, dtype=np.float32)
    y[0] = 1.0
    sf.write(src / "a.wav", y, sr)

    dst = tmp_path / "dst"
    data_augmentation.augment_wav_dir(src, dst, [0, 1], [1.0], [20])

    files = list(dst.rglob("*.wav"))
    assert len(files) == 2
    for f in files:
        data, sr2 = sf.read(f)
        assert abs(len(data) - len(y)) <= sr * 0.1

