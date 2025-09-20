import os
import wave
from pathlib import Path

import numpy as np
import pretty_midi as pm

from utilities.drumstem_to_midi import Options, convert

SR = 44100


def write_wav(path: Path, samples: np.ndarray, sr: int = SR) -> None:
    data = np.clip(samples * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(data.tobytes())


def synth_drums_like(duration: float = 4.0) -> np.ndarray:
    np.random.seed(0)
    total = int(SR * duration)
    x = np.zeros(total, dtype=np.float32)
    frame_kick = int(0.01 * SR)
    frame_snare = int(0.008 * SR)
    frame_hat = int(0.004 * SR)
    hop = int(0.5 * SR)
    for start in range(int(0.2 * SR), total - hop, hop):
        x[start : start + frame_kick] += 0.9 * np.hanning(frame_kick)
        sn = start + int(0.25 * SR)
        sn_end = sn + frame_snare
        noise_sn = np.random.rand(frame_snare).astype(np.float32) * 2 - 1
        x[sn:sn_end] += 0.5 * noise_sn
        hh = start + int(0.125 * SR)
        noise_hh = np.random.rand(frame_hat).astype(np.float32) * 2 - 1
        filt = np.ones(32, dtype=np.float32) / 32.0
        noise_hh = noise_hh - np.convolve(noise_hh, filt, mode="same")
        x[hh : hh + frame_hat] += 0.4 * noise_hh
    peak = np.max(np.abs(x))
    if peak > 0:
        x /= peak
    return x


def test_stem_to_midi(tmp_path):
    wav_path = tmp_path / "stem.wav"
    midi_path = tmp_path / "out.mid"
    samples = synth_drums_like()
    write_wav(wav_path, samples)

    np.random.seed(1)
    opt = Options(
        bpm=120.0,
        sr=SR,
        gate_db=-80.0,
        tight=0.6,
        min_sep_ms=25.0,
        humanize_ms=1.5,
        kick_note=36,
        snare_note=38,
        hihat_note=42,
        win_ms=20.0,
        hop_ms=5.0,
    )
    convert(str(wav_path), str(midi_path), opt)

    assert os.path.exists(midi_path)
    midi = pm.PrettyMIDI(str(midi_path))
    assert midi.instruments
    inst = midi.instruments[0]
    assert inst.is_drum is True
    assert inst.notes
    assert all(1 <= note.velocity <= 127 for note in inst.notes)
