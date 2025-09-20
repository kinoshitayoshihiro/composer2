import wave
from pathlib import Path

import numpy as np

from utilities.tempo_from_mix import (
    estimate_bpm,
    load_mono,
    onset_envelope,
    track_beats,
)

SR = 44100

def write_wav(path: Path, samples: np.ndarray, sr: int = SR) -> None:
    data = np.clip(samples * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(data.tobytes())


def test_click_120bpm_estimation(tmp_path):
    duration = 10.0
    total = int(SR * duration)
    x = np.zeros(total, dtype=np.float32)
    hop = int(0.5 * SR)
    pulse = int(0.002 * SR)
    for idx in range(0, total, hop):
        x[idx : idx + pulse] = 1.0
    wav_path = tmp_path / "click.wav"
    write_wav(wav_path, x)

    signal, sr = load_mono(str(wav_path), SR)
    envelope, hop_s = onset_envelope(signal, sr, 20.0, 5.0)
    bpm = estimate_bpm(envelope, hop_s, 60.0, 200.0)
    assert 116 <= bpm <= 124, bpm
    beats = track_beats(envelope, hop_s, bpm)
    assert len(beats) > 8
    assert all(beats[i] < beats[i + 1] for i in range(len(beats) - 1))
