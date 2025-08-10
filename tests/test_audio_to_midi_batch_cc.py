import importlib.util
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pretty_midi = pytest.importorskip("pretty_midi")

spec = importlib.util.spec_from_file_location(
    "utilities.audio_to_midi_batch",
    Path(__file__).resolve().parents[1] / "utilities" / "audio_to_midi_batch.py",
)
audio_to_midi_batch = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = audio_to_midi_batch
spec.loader.exec_module(audio_to_midi_batch)
StemResult = audio_to_midi_batch.StemResult


def _write(path: Path, data: np.ndarray, sr: int) -> None:
    import wave

    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes((data * 32767).astype("<i2").tobytes())


def _stub_transcribe(path: Path, *, cc_strategy: str = "none", cc11_smoothing_ms: int = 80, sustain_threshold: float = 0.0, **kwargs) -> StemResult:
    inst = pretty_midi.Instrument(program=0, name=path.stem)
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=0.3))
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=64, start=0.35, end=0.6))
    sr = 1000
    audio = np.random.rand(int(sr * 0.6))
    if cc_strategy != "none":
        events = audio_to_midi_batch.cc_utils.energy_to_cc11(
            audio, sr, smooth_ms=cc11_smoothing_ms, strategy=cc_strategy
        )
        prev = None
        for t, v in events:
            if prev is None or v != prev:
                inst.control_changes.append(pretty_midi.ControlChange(number=11, value=v, time=float(t)))
                prev = v
    if sustain_threshold > 0 and "piano" in path.stem.lower():
        for t, v in audio_to_midi_batch.cc_utils.infer_cc64_from_overlaps(inst.notes, sustain_threshold):
            inst.control_changes.append(pretty_midi.ControlChange(number=64, value=v, time=float(t)))
    return StemResult(inst, 120.0)


def test_cc11_smoothing(tmp_path, monkeypatch):
    in_dir = tmp_path / "wav"
    out0 = tmp_path / "out0"
    out1 = tmp_path / "out1"
    in_dir.mkdir()
    sr = 22050
    t = np.linspace(0, 1, sr, False)
    wave = 0.1 * np.sin(2 * np.pi * 440 * t)
    _write(in_dir / "sample.wav", wave, sr)
    monkeypatch.setattr(audio_to_midi_batch, "_transcribe_stem", _stub_transcribe)
    audio_to_midi_batch.main([str(in_dir), str(out0), "--cc-strategy", "energy", "--cc11-smoothing-ms", "0"])
    audio_to_midi_batch.main([str(in_dir), str(out1), "--cc-strategy", "energy", "--cc11-smoothing-ms", "200"])
    pm0 = pretty_midi.PrettyMIDI(str(out0 / in_dir.name / "sample.mid"))
    pm1 = pretty_midi.PrettyMIDI(str(out1 / in_dir.name / "sample.mid"))
    cc0 = [cc.value for cc in pm0.instruments[0].control_changes if cc.number == 11]
    cc1 = [cc.value for cc in pm1.instruments[0].control_changes if cc.number == 11]
    assert all(0 <= v <= 127 for v in cc0)
    assert cc1 and len(cc1) < len(cc0)


def test_sustain_generation(tmp_path, monkeypatch):
    in_dir = tmp_path / "wav"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    sr = 22050
    t = np.linspace(0, 1, sr, False)
    wave = 0.1 * np.sin(2 * np.pi * 440 * t)
    _write(in_dir / "piano.wav", wave, sr)
    monkeypatch.setattr(audio_to_midi_batch, "_transcribe_stem", _stub_transcribe)
    audio_to_midi_batch.main([
        str(in_dir),
        str(out_dir),
        "--cc-strategy",
        "none",
        "--sustain-threshold",
        "0.1",
    ])
    pm = pretty_midi.PrettyMIDI(str(out_dir / in_dir.name / "piano.mid"))
    cc64 = [cc.value for cc in pm.instruments[0].control_changes if cc.number == 64]
    assert cc64 == [127, 0]


def test_cc11_without_librosa(tmp_path, monkeypatch):
    in_dir = tmp_path / "wav"
    in_dir.mkdir()
    sr = 8000
    t = np.linspace(0, 0.5, int(sr * 0.5), False)
    wave = 0.1 * np.sin(2 * np.pi * 440 * t)
    path = in_dir / "sample.wav"
    _write(path, wave, sr)
    monkeypatch.setattr(audio_to_midi_batch, "librosa", None)
    monkeypatch.setattr(audio_to_midi_batch, "crepe", None)
    res = audio_to_midi_batch._transcribe_stem(
        path,
        cc_strategy="energy",
        cc11_smoothing_ms=0,
        sustain_threshold=0.0,
        auto_tempo=False,
    )
    cc11 = [cc for cc in res.instrument.control_changes if cc.number == 11]
    assert cc11


def test_no_sustain_when_threshold_zero(tmp_path, monkeypatch):
    in_dir = tmp_path / "wav"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    sr = 22050
    t = np.linspace(0, 1, sr, False)
    wave = 0.1 * np.sin(2 * np.pi * 440 * t)
    _write(in_dir / "piano.wav", wave, sr)
    monkeypatch.setattr(audio_to_midi_batch, "_transcribe_stem", _stub_transcribe)
    audio_to_midi_batch.main(
        [str(in_dir), str(out_dir), "--cc-strategy", "none", "--sustain-threshold", "0"]
    )
    pm = pretty_midi.PrettyMIDI(str(out_dir / in_dir.name / "piano.mid"))
    cc64 = [cc for cc in pm.instruments[0].control_changes if cc.number == 64]
    assert not cc64


def test_cc64_threshold_alias(tmp_path, monkeypatch):
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    params: dict[str, float] = {}

    def fake_convert(_, __, *, sustain_threshold, **kwargs):
        params["sustain_threshold"] = sustain_threshold

    monkeypatch.setattr(audio_to_midi_batch, "convert_directory", fake_convert)
    with pytest.warns(DeprecationWarning):
        audio_to_midi_batch.main([
            str(in_dir),
            str(out_dir),
            "--cc64-threshold",
            "0.5",
        ])
    assert params.get("sustain_threshold") == pytest.approx(0.5)
