import numpy as np
import soundfile as sf
from pathlib import Path

from utilities import audio_analysis


def test_silence_envelope(tmp_path: Path) -> None:
    path = tmp_path / "silence.wav"
    sr = 22050
    sf.write(path, np.zeros(sr), sr)
    env = audio_analysis.extract_amplitude_envelope(path)
    assert all(v == 0 for v in env)
    vel = audio_analysis.map_envelope_to_velocity(env, min_vel=30)
    assert vel == [30] * len(env)


def test_pitch_shift_detection(tmp_path: Path) -> None:
    sr = 22050
    t = np.linspace(0, 1.0, sr, endpoint=False)
    tone = np.sin(2 * np.pi * 440 * t)
    path = tmp_path / "tone.wav"
    sf.write(path, tone, sr)
    shifted = audio_analysis.extract_pitch(path)
    base = float(np.nanmean(shifted))

    # pitch shift +2 semitones
    from utilities.data_augmentation import pitch_shift

    shifted_audio = pitch_shift(tone, sr, 2)
    shifted_path = tmp_path / "tone_ps.wav"
    sf.write(shifted_path, shifted_audio, sr)
    shifted_pitch = audio_analysis.extract_pitch(shifted_path)
    shifted_mean = float(np.nanmean(shifted_pitch))

    expected = base * 2 ** (2 / 12)
    assert abs(shifted_mean - expected) < 10

