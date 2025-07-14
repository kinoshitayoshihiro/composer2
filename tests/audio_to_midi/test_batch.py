from pathlib import Path

import numpy as np
import pretty_midi
import soundfile as sf
import sys
import types

sys.modules.setdefault("basic_pitch", types.ModuleType("basic_pitch"))
sys.modules["basic_pitch"].inference = types.ModuleType("inference")
from basic_pitch.inference import predict  # noqa: F401
from utilities import audio_to_midi_batch


def fake_predict(path: str, *_, **__):
    return {}, pretty_midi.PrettyMIDI(), [(0.0, 0.1, 45, 0.2, None)]


def test_audio_to_midi_batch(tmp_path, monkeypatch):
    in_dir = tmp_path / "wav"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    sr = 22050
    t = np.linspace(0, 1, sr, False)
    wave = 0.1 * np.sin(2 * np.pi * 440 * t)
    for i in range(3):
        sf.write(in_dir / f"sample{i}.wav", wave, sr)

    monkeypatch.setattr("basic_pitch.inference.predict", fake_predict)
    audio_to_midi_batch.main([str(in_dir), str(out_dir), "--jobs", "1"])
    mids = list(out_dir.glob("*.mid"))
    assert len(mids) == 3
    pm = pretty_midi.PrettyMIDI(str(mids[0]))
    assert any(n.pitch == 36 for n in pm.instruments[0].notes)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
