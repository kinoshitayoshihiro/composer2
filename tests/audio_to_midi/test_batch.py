import sys
import types
from pathlib import Path

from pathlib import Path
import sys
import types

import pytest
np = pytest.importorskip("numpy")
pretty_midi = pytest.importorskip("pretty_midi")
import wave

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
from utilities.audio_to_midi_batch import StemResult


def fake_predict(path: str, *args, **kwargs):
    """Alternative fake predict function for monkeypatch"""
    return {}, pretty_midi.PrettyMIDI(), [(0.0, 0.1, 45, 0.2, None)]


def _write(path: Path, data: np.ndarray, sr: int) -> None:
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes((data * 32767).astype("<i2").tobytes())


def test_audio_to_midi_batch(tmp_path, monkeypatch):
    """Test audio to MIDI batch processing with mocked basic_pitch"""
    in_dir = tmp_path / "wav"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    sr = 22050
    t = np.linspace(0, 1, sr, False)
    wave = 0.1 * np.sin(2 * np.pi * 440 * t)
    for i in range(3):
        _write(in_dir / f"sample{i}.wav", wave, sr)

    monkeypatch.setattr("basic_pitch.inference.predict", fake_predict)
    audio_to_midi_batch.main([str(in_dir), str(out_dir), "--jobs", "1"])
    midi_dir = out_dir / in_dir.name
    mids = list(midi_dir.glob("*.mid"))
    assert len(mids) == 3
    pm = pretty_midi.PrettyMIDI(str(mids[0]))
    assert any(n.pitch == 36 for n in pm.instruments[0].notes)


def _pitch_stub(pitch: int):
    def _stub(
        path: Path,
        *,
        step_size: int = 10,
        conf_threshold: float = 0.5,
        min_dur: float = 0.05,
        auto_tempo: bool = True,
    ) -> StemResult:
        inst = pretty_midi.Instrument(program=0, name=path.stem)
        inst.notes.append(
            pretty_midi.Note(velocity=100, pitch=pitch, start=0.0, end=min_dur)
        )
        return StemResult(inst, 120.0)

    return _stub


def test_resume_new_stems(tmp_path, monkeypatch):
    song_dir = tmp_path / "song"
    out_dir = tmp_path / "out"
    song_dir.mkdir()
    sr = 22050
    t = np.linspace(0, 1, sr, False)
    wave = 0.1 * np.sin(2 * np.pi * 440 * t)
    _write(song_dir / "a.wav", wave, sr)

    monkeypatch.setattr(audio_to_midi_batch, "_transcribe_stem", _pitch_stub(60))
    audio_to_midi_batch.main([str(song_dir), str(out_dir), "--resume"])

    _write(song_dir / "b.wav", wave, sr)
    monkeypatch.setattr(audio_to_midi_batch, "_transcribe_stem", _pitch_stub(61))
    audio_to_midi_batch.main([str(song_dir), str(out_dir), "--resume"])

    pm_a = pretty_midi.PrettyMIDI(str(out_dir / song_dir.name / "a.mid"))
    pm_b = pretty_midi.PrettyMIDI(str(out_dir / song_dir.name / "b.mid"))
    assert pm_a.instruments[0].notes[0].pitch == 60
    assert pm_b.instruments[0].notes[0].pitch == 61


def test_overwrite(tmp_path, monkeypatch):
    song_dir = tmp_path / "song"
    out_dir = tmp_path / "out"
    song_dir.mkdir()
    sr = 22050
    t = np.linspace(0, 1, sr, False)
    wave = 0.1 * np.sin(2 * np.pi * 440 * t)
    _write(song_dir / "a.wav", wave, sr)

    monkeypatch.setattr(audio_to_midi_batch, "_transcribe_stem", _pitch_stub(60))
    audio_to_midi_batch.main([str(song_dir), str(out_dir)])

    monkeypatch.setattr(audio_to_midi_batch, "_transcribe_stem", _pitch_stub(61))
    audio_to_midi_batch.main([str(song_dir), str(out_dir), "--overwrite"])

    midi_dir = out_dir / song_dir.name
    mids = list(midi_dir.glob("*.mid"))
    assert {p.name for p in mids} == {"a.mid"}
    pm = pretty_midi.PrettyMIDI(str(mids[0]))
    assert pm.instruments[0].notes[0].pitch == 61


def test_collision_renaming(tmp_path, monkeypatch):
    song_dir = tmp_path / "song"
    out_dir = tmp_path / "out"
    song_dir.mkdir()
    sr = 22050
    t = np.linspace(0, 1, sr, False)
    wave = 0.1 * np.sin(2 * np.pi * 440 * t)
    _write(song_dir / "a.wav", wave, sr)
    _write(song_dir / "a!.wav", wave, sr)

    monkeypatch.setattr(audio_to_midi_batch, "_transcribe_stem", _pitch_stub(60))
    audio_to_midi_batch.main([str(song_dir), str(out_dir)])

    midi_dir = out_dir / song_dir.name
    assert {p.name for p in midi_dir.glob("*.mid")} == {"a.mid", "a_1.mid"}


def test_safe_dirnames(tmp_path, monkeypatch):
    src_root = tmp_path / "songs"
    src_root.mkdir()
    song_dir = src_root / "Song! 1"
    song_dir.mkdir()
    out_dir = tmp_path / "out"
    sr = 22050
    t = np.linspace(0, 1, sr, False)
    wave = 0.1 * np.sin(2 * np.pi * 440 * t)
    _write(song_dir / "a.wav", wave, sr)

    monkeypatch.setattr(audio_to_midi_batch, "_transcribe_stem", _pitch_stub(60))
    audio_to_midi_batch.main([str(src_root), str(out_dir), "--safe-dirnames"])

    assert (out_dir / "Song_1" / "a.mid").exists()
