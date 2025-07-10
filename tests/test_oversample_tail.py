import numpy as np
import soundfile as sf
from utilities.convolver import render_with_ir


def test_oversample_tail(tmp_path):
    sr = 44100
    dur = 0.25
    audio = np.zeros(int(sr * dur), dtype=np.float32)
    audio[0] = 1.0
    ir = np.exp(-np.linspace(0, 1.0, int(sr * dur))).astype(np.float32)
    wav = tmp_path / "dry.wav"
    irwav = tmp_path / "ir.wav"
    out = tmp_path / "out.wav"
    sf.write(wav, audio, sr)
    sf.write(irwav, ir, sr)
    render_with_ir(wav, irwav, out, oversample=2, normalize=False, tail_db_drop=-40)
    data, _ = sf.read(out)
    assert len(data) > len(audio)
    tail_db = 20 * np.log10(abs(data[-1]) / max(abs(data)))
    assert tail_db < -40
    peak = max(abs(data))
    thresh = peak * (10 ** (-40 / 20.0))
    idx = np.where(np.abs(data) > thresh)[0]
    start = idx[-1] if idx.size else 0
    fade_len = len(data) - start
    assert abs(fade_len - int(sr * 0.01)) <= 2
