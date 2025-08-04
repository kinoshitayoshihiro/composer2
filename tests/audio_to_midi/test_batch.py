import sys
import types
from pathlib import Path

import numpy as np
import pretty_midi
import soundfile as sf

# Mock basic_pitch module before any imports that might use it
basic_pitch_module = types.ModuleType("basic_pitch")
inference_module = types.ModuleType("inference")


def mock_predict(path: str, *args, **kwargs):
    """Mock predict function for testing without basic_pitch dependency"""
    return {}, pretty_midi.PrettyMIDI(), [(0.0, 0.1, 45, 0.2, None)]


# Set up the mock modules
inference_module.predict = mock_predict
basic_pitch_module.inference = inference_module

sys.modules["basic_pitch"] = basic_pitch_module
sys.modules["basic_pitch.inference"] = inference_module

# Now safe to import modules that depend on basic_pitch
from utilities import audio_to_midi_batch


def fake_predict(path: str, *args, **kwargs):
    """Alternative fake predict function for monkeypatch"""
    return {}, pretty_midi.PrettyMIDI(), [(0.0, 0.1, 45, 0.2, None)]


def test_audio_to_midi_batch(tmp_path, monkeypatch):
    """Test audio to MIDI batch processing with mocked basic_pitch"""
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
    assert len(mids) == 1
    pm = pretty_midi.PrettyMIDI(str(mids[0]))
    assert any(n.pitch == 36 for n in pm.instruments[0].notes)
