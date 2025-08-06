import hashlib
import sys
import types
from pathlib import Path

import numpy as np
import pretty_midi
import pytest
import soundfile as sf

from utilities import audio_to_midi_batch


@pytest.fixture(autouse=True)
def _no_tqdm(monkeypatch):
    class DummyTqdm:
        def __init__(self, iterable=None, **kwargs):
            self.iterable = iterable

        def __iter__(self):
            if self.iterable is None:
                return iter([])
            return iter(self.iterable)

        def update(self, n=1):
            pass

        def close(self):
            pass

    def dummy(iterable=None, **kwargs):
        return DummyTqdm(iterable)

    monkeypatch.setattr(audio_to_midi_batch, "tqdm", dummy)


def _stub_transcribe(
    path: Path,
    *,
    step_size: int = 10,
    conf_threshold: float = 0.5,
    min_dur: float = 0.05,
) -> pretty_midi.Instrument:
    inst = pretty_midi.Instrument(program=0, name=path.stem)
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=min_dur))
    return inst


def test_fallback_basic_pitch(tmp_path, monkeypatch):
    in_dir = tmp_path / "wav"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    sr = 22050
    t = np.linspace(0, 1, sr, False)
    wave = 0.1 * np.sin(2 * np.pi * 440 * t)
    sf.write(in_dir / "sample.wav", wave, sr)

    basic_pitch_module = types.ModuleType("basic_pitch")
    inference_module = types.ModuleType("inference")

    def fake_predict(path: str, *args, **kwargs):
        return {}, pretty_midi.PrettyMIDI(), [(0.0, 0.2, 45, 0.1, None)]

    inference_module.predict = fake_predict
    basic_pitch_module.inference = inference_module
    sys.modules["basic_pitch"] = basic_pitch_module
    sys.modules["basic_pitch.inference"] = inference_module

    monkeypatch.setattr(audio_to_midi_batch, "crepe", None)
    monkeypatch.setattr(audio_to_midi_batch, "librosa", None)

    audio_to_midi_batch.main([str(in_dir), str(out_dir)])
    midi_dir = out_dir / in_dir.name
    mids = list(midi_dir.glob("*.mid"))
    assert len(mids) == 1
    pm = pretty_midi.PrettyMIDI(str(mids[0]))
    assert pm.instruments[0].notes[0].pitch == 36


def test_parallel_jobs_and_cli(tmp_path, monkeypatch):
    in_dir = tmp_path / "song"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    sr = 22050
    t = np.linspace(0, 1, sr, False)
    wave = 0.1 * np.sin(2 * np.pi * 440 * t)
    sf.write(in_dir / "a.wav", wave, sr)
    sf.write(in_dir / "b.wav", wave, sr)

    monkeypatch.setattr(audio_to_midi_batch, "_transcribe_stem", _stub_transcribe)

    audio_to_midi_batch.main(
        [
            str(in_dir),
            str(out_dir),
            "--jobs",
            "2",
            "--ext",
            "wav",
            "--min-dur",
            "0.1",
            "--merge",
        ]
    )
    midi_path = out_dir / f"{in_dir.name}.mid"
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    assert len(pm.instruments) == 2
    assert pm.instruments[0].notes[0].end == pytest.approx(0.1)


def test_multi_ext_scanning(tmp_path, monkeypatch):
    song_dir = tmp_path / "song"
    out_dir = tmp_path / "out"
    song_dir.mkdir()
    sr = 22050
    t = np.linspace(0, 1, sr, False)
    wave = 0.1 * np.sin(2 * np.pi * 440 * t)
    sf.write(song_dir / "a.wav", wave, sr)
    sf.write(song_dir / "b.flac", wave, sr)

    monkeypatch.setattr(audio_to_midi_batch, "_transcribe_stem", _stub_transcribe)

    audio_to_midi_batch.main([str(song_dir), str(out_dir), "--ext", "wav,flac"])
    midi_dir = out_dir / song_dir.name
    assert {p.stem for p in midi_dir.glob("*.mid")} == {"a", "b"}


def test_resume_skips_existing(tmp_path, monkeypatch):
    song_dir = tmp_path / "song"
    out_dir = tmp_path / "out"
    song_dir.mkdir()
    sr = 22050
    t = np.linspace(0, 1, sr, False)
    wave = 0.1 * np.sin(2 * np.pi * 440 * t)
    sf.write(song_dir / "a.wav", wave, sr)

    monkeypatch.setattr(audio_to_midi_batch, "_transcribe_stem", _stub_transcribe)

    audio_to_midi_batch.main([str(song_dir), str(out_dir), "--resume"])
    midi_path = out_dir / song_dir.name / "a.mid"
    assert midi_path.exists()

    def fail_transcribe(path: Path, **kwargs):  # pragma: no cover - ensure skip
        raise AssertionError("should not transcribe on resume")

    monkeypatch.setattr(audio_to_midi_batch, "_transcribe_stem", fail_transcribe)
    audio_to_midi_batch.main([str(song_dir), str(out_dir), "--resume"])


def _pitch_stub(pitch: int):
    def _stub(
        path: Path,
        *,
        step_size: int = 10,
        conf_threshold: float = 0.5,
        min_dur: float = 0.05,
    ) -> pretty_midi.Instrument:
        inst = pretty_midi.Instrument(program=0, name=path.stem)
        inst.notes.append(
            pretty_midi.Note(velocity=100, pitch=pitch, start=0.0, end=min_dur)
        )
        return inst

    return _stub


def test_resume_new_stems(tmp_path, monkeypatch):
    song_dir = tmp_path / "song"
    out_dir = tmp_path / "out"
    song_dir.mkdir()
    sr = 22050
    t = np.linspace(0, 1, sr, False)
    wave = 0.1 * np.sin(2 * np.pi * 440 * t)
    sf.write(song_dir / "a.wav", wave, sr)

    monkeypatch.setattr(audio_to_midi_batch, "_transcribe_stem", _pitch_stub(60))
    audio_to_midi_batch.main([str(song_dir), str(out_dir), "--resume"])

    sf.write(song_dir / "b.wav", wave, sr)
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
    sf.write(song_dir / "a.wav", wave, sr)

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
    sf.write(song_dir / "a.wav", wave, sr)
    sf.write(song_dir / "a!.wav", wave, sr)

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
    sf.write(song_dir / "a.wav", wave, sr)

    monkeypatch.setattr(audio_to_midi_batch, "_transcribe_stem", _pitch_stub(60))
    audio_to_midi_batch.main([str(src_root), str(out_dir), "--safe-dirnames"])

    assert (out_dir / "Song_1" / "a.mid").exists()


def test_safe_dirnames_collision(tmp_path, monkeypatch):
    src_root = tmp_path / "songs"
    src_root.mkdir()
    song_a = src_root / "Song! 1"
    song_b = src_root / "Song? 1"
    song_a.mkdir()
    song_b.mkdir()
    out_dir = tmp_path / "out"
    sr = 22050
    t = np.linspace(0, 1, sr, False)
    wave = 0.1 * np.sin(2 * np.pi * 440 * t)
    sf.write(song_a / "a.wav", wave, sr)
    sf.write(song_b / "a.wav", wave, sr)

    monkeypatch.setattr(audio_to_midi_batch, "_transcribe_stem", _pitch_stub(60))
    audio_to_midi_batch.main([str(src_root), str(out_dir), "--safe-dirnames"])

    base = out_dir / "Song_1"
    assert base.exists()
    h1 = hashlib.sha1("Song! 1".encode()).hexdigest()[:8]
    h2 = hashlib.sha1("Song? 1".encode()).hexdigest()[:8]
    hashed_dirs = [out_dir / f"Song_1__{h1}", out_dir / f"Song_1__{h2}"]
    assert sum(p.exists() for p in hashed_dirs) == 1
    assert any((p / "a.mid").exists() for p in [base] + hashed_dirs)
