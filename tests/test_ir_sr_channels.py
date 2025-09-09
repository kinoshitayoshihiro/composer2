import numpy as np
import pytest
import soundfile as sf

from utilities.convolver import render_with_ir

pytestmark = pytest.mark.requires_audio


@pytest.mark.parametrize("sr", [44100, 96000, 192000])
@pytest.mark.parametrize("channels", [1, 2, 3, 4, 5])
def test_ir_sr_and_channels(tmp_path, monkeypatch, sr, channels):
    import utilities.convolver as conv

    monkeypatch.setattr(conv, "soxr", None)

    audio = np.zeros(int(sr * 0.01), dtype=np.float32)
    audio[0] = 1.0
    inp = tmp_path / "in.wav"
    sf.write(inp, audio, sr)

    ir = np.zeros((int(sr * 0.01), channels), dtype=np.float32)
    ir[0] = 1.0
    ir_path = tmp_path / "ir.wav"
    sf.write(ir_path, ir, sr)

    out = tmp_path / "out.wav"
    render_with_ir(inp, ir_path, out, normalize=False)

    data, out_sr = sf.read(out, always_2d=True)
    assert out_sr == 44100
    expected_ch = 1 if channels == 1 else 2
    assert data.shape[1] == expected_ch


@pytest.mark.parametrize("sr", [44100, 96000, 192000])
def test_downmix_none_preserves_channels(tmp_path, monkeypatch, sr):
    import utilities.convolver as conv

    monkeypatch.setattr(conv, "soxr", None)

    audio = np.zeros(int(sr * 0.01), dtype=np.float32)
    audio[0] = 1.0
    inp = tmp_path / "in.wav"
    sf.write(inp, audio, sr)

    ir = np.zeros((int(sr * 0.01), 4), dtype=np.float32)
    ir[0] = 1.0
    ir_path = tmp_path / "ir.wav"
    sf.write(ir_path, ir, sr)

    out = tmp_path / "out.wav"
    render_with_ir(inp, ir_path, out, normalize=False, downmix="none")

    data, out_sr = sf.read(out, always_2d=True)
    assert out_sr == 44100
    assert data.shape[1] == 4


@pytest.mark.parametrize("channels", [3, 5])
def test_mix_to_stereo_rms_energy(channels):
    import utilities.convolver as conv

    ir = np.random.random((128, channels)).astype(np.float32)
    stereo = conv._mix_to_stereo(ir)

    assert stereo.shape == (128, 2)
    orig_pow = float(np.sum(ir**2))
    new_pow = float(np.sum(stereo**2))
    assert np.allclose(new_pow, orig_pow)
