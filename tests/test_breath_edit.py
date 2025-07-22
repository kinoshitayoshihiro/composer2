from __future__ import annotations

import contextlib
import importlib.util
import time
import wave
from pathlib import Path

import numpy as np
import pytest


def write_wav(path: Path, data: np.ndarray, sr: int) -> None:
    arr = (np.clip(data, -1.0, 1.0) * 32767).astype(np.int16)
    with contextlib.closing(wave.open(str(path), "wb")) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(arr.tobytes())


spec = importlib.util.spec_from_file_location(
    "breath_edit",
    (Path(__file__).resolve().parents[1] / "utilities/breath_edit.py"),
)
breath_edit = importlib.util.module_from_spec(spec)
assert spec.loader
spec.loader.exec_module(breath_edit)
process_breath = breath_edit.process_breath

spec_mask = importlib.util.spec_from_file_location(
    "breath_mask",
    (Path(__file__).resolve().parents[1] / "utilities/breath_mask.py"),
)
breath_mask = importlib.util.module_from_spec(spec_mask)
assert spec_mask.loader
spec_mask.loader.exec_module(breath_mask)
infer_breath_mask = breath_mask.infer_breath_mask


@pytest.mark.parametrize("mode", ["keep", "attenuate", "remove"])
def test_process_breath(tmp_path: Path, mode: str) -> None:
    pd = pytest.importorskip("pydub")
    from pydub import AudioSegment

    pydub_version = getattr(pd, "__version__", "0")
    sr = 16000
    t = np.arange(sr * 1.2) / sr
    sine = 0.3 * np.sin(2 * np.pi * 1000 * t)
    noise = sine[: sr // 2]
    breath = sine[sr // 2 : sr // 2 + int(sr * 0.2)]
    tail = sine[sr // 2 + int(sr * 0.2) :]
    samples = np.concatenate([noise, breath, tail])
    inp = tmp_path / "in.wav"
    write_wav(inp, samples, sr)

    mask = np.zeros(len(samples) // 160, dtype=bool)
    mask[len(noise) // 160 : (len(noise) + len(breath)) // 160] = True

    out = tmp_path / "out.wav"
    process_breath(inp, out, mask, mode, -15.0, 50, hop_ms=10)

    audio = AudioSegment.from_file(out)
    total_ms = len(samples) / sr * 1000
    if mode == "keep":
        assert abs(len(audio) - total_ms) <= 2
        ref = AudioSegment.from_file(inp)
        diff = 20 * np.log10((audio.rms or 1) / (ref.rms or 1))
        assert abs(diff) < 1
    elif mode == "attenuate":
        start = int(len(noise) / sr * 1000)
        seg = audio[start : start + 200]
        ref = audio[start - 200 : start]
        drop = 20 * np.log10((seg.rms or 1) / (ref.rms or 1))
        assert abs(drop - (-15.0)) <= 2.0
        assert abs(len(audio) - total_ms) <= 2
    else:
        ver = tuple(int(v) for v in pydub_version.split(".")[:2])
        xfade = 50 if ver < (0, 25) else 0
        expected = len(noise) / sr * 1000 + len(tail) / sr * 1000 - xfade
        assert abs(len(audio) - expected) <= 5


def test_process_breath_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("pydub")
    sr = 16000
    samples = np.random.randn(sr * 120).astype(np.float32)
    inp = tmp_path / "in.wav"
    write_wav(inp, samples, sr)
    mask = np.zeros(int(len(samples) / (sr * 0.01)), dtype=bool)
    out = tmp_path / "out.wav"
    start = time.perf_counter()
    monkeypatch.setattr(breath_edit, "pydub_version", "0.24.1")
    process_breath(inp, out, mask, "remove", -15.0, 0, hop_ms=10)
    elapsed = time.perf_counter() - start
    assert elapsed < 4.0


def test_micro_segment_fade(tmp_path: Path) -> None:
    pytest.importorskip("pydub")
    from pydub import AudioSegment

    sr = 16000
    noise = 0.3 * np.random.randn(int(sr * 0.05)).astype(np.float32)
    breath = np.zeros(int(sr * 0.01), dtype=np.float32)
    samples = np.concatenate([noise, breath, noise])
    inp = tmp_path / "in.wav"
    write_wav(inp, samples, sr)
    mask = np.zeros(len(samples) // 160, dtype=bool)
    start = len(noise) // 160
    mask[start : start + len(breath) // 160] = True
    out = tmp_path / "out.wav"
    process_breath(inp, out, mask, "remove", -15.0, 50, hop_ms=10)
    audio = AudioSegment.from_file(out)
    join_ms = len(noise) / sr * 1000
    pre = audio[join_ms - 5 : join_ms]
    post = audio[join_ms : join_ms + 5]
    diff = 20 * np.log10((post.rms or 1) / (pre.rms or 1))
    assert abs(diff) < 3


def test_infer_breath_mask_percentile(tmp_path: Path) -> None:
    pytest.importorskip("soundfile")
    import soundfile as sf

    sr = 16000
    low = 0.001 * np.random.randn(sr).astype(np.float32)
    loud = 0.5 * np.random.randn(sr).astype(np.float32)
    samples = np.concatenate([low, loud])
    wav = tmp_path / "in.wav"
    sf.write(wav, samples, sr)
    mask = infer_breath_mask(wav, hop_ms=10, thr_offset_db=-30, percentile=80)
    assert len(mask) > 0


def test_infer_breath_mask_onnx_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    pytest.importorskip("soundfile")
    import soundfile as sf

    sr = 16000
    samples = np.random.randn(sr).astype(np.float32)
    wav = tmp_path / "a.wav"
    sf.write(wav, samples, sr)
    model = tmp_path / "m.onnx"
    model.write_bytes(b"x")

    class DummySession:
        def __init__(self, path: str) -> None:
            self.path = path

    dummy = type("D", (), {"InferenceSession": DummySession})
    monkeypatch.setattr(breath_mask, "ort", dummy)
    monkeypatch.setenv("BREATH_ONNX_PATH", str(model))
    mask = infer_breath_mask(wav, hop_ms=10)
    assert isinstance(mask, np.ndarray)
    assert model.exists()

    monkeypatch.setenv("BREATH_ONNX_PATH", str(tmp_path / "missing.onnx"))
    with caplog.at_level("WARNING"):
        infer_breath_mask(wav, hop_ms=10)
    assert "falling back" in caplog.text
