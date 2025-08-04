import sys
import types
from pathlib import Path

import numpy as np
import pretty_midi
import pytest
import soundfile as sf

from utilities import audio_to_midi_batch


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
    mids = list(out_dir.glob("*.mid"))
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
        [str(in_dir), str(out_dir), "--jobs", "2", "--ext", "wav", "--min-dur", "0.1"]
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
    midi_path = out_dir / f"{song_dir.name}.mid"
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    assert {inst.name for inst in pm.instruments} == {"a", "b"}


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
    midi_path = out_dir / f"{song_dir.name}.mid"
    assert midi_path.exists()

    def fail_transcribe(path: Path, **kwargs):  # pragma: no cover - ensure skip
        raise AssertionError("should not transcribe on resume")

    monkeypatch.setattr(audio_to_midi_batch, "_transcribe_stem", fail_transcribe)
    audio_to_midi_batch.main([str(song_dir), str(out_dir), "--resume"])
