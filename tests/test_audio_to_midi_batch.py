import sys
import types
from pathlib import Path

import numpy as np
import sys
import types
from pathlib import Path

import numpy as np
import pretty_midi
import pytest
import wave
import multiprocessing

from utilities import audio_to_midi_batch
from utilities.audio_to_midi_batch import StemResult


def _stub_transcribe(
    path: Path,
    *,
    step_size: int = 10,
    conf_threshold: float = 0.5,
    min_dur: float = 0.05,
    auto_tempo: bool = True,
) -> StemResult:
    inst = pretty_midi.Instrument(program=0, name=path.stem)
    inst.notes.append(
        pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=min_dur)
    )
    return StemResult(inst, 120.0)


def _write(path: Path, data: np.ndarray, sr: int) -> None:
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes((data * 32767).astype("<i2").tobytes())


def test_fallback_basic_pitch(tmp_path, monkeypatch):
    in_dir = tmp_path / "wav"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    sr = 22050
    t = np.linspace(0, 1, sr, False)
    wave = 0.1 * np.sin(2 * np.pi * 440 * t)
    _write(in_dir / "sample.wav", wave, sr)

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
    _write(in_dir / "a.wav", wave, sr)
    _write(in_dir / "b.wav", wave, sr)

    monkeypatch.setattr(audio_to_midi_batch, "_transcribe_stem", _stub_transcribe)
    orig_ctx = multiprocessing.get_context
    monkeypatch.setattr(
        audio_to_midi_batch.multiprocessing,
        "get_context",
        lambda method: orig_ctx("fork"),
    )

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
    _write(song_dir / "a.wav", wave, sr)
    _write(song_dir / "b.flac", wave, sr)

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
    _write(song_dir / "a.wav", wave, sr)

    monkeypatch.setattr(audio_to_midi_batch, "_transcribe_stem", _stub_transcribe)

    audio_to_midi_batch.main([str(song_dir), str(out_dir), "--resume"])
    midi_path = out_dir / song_dir.name / "a.mid"
    assert midi_path.exists()

    def fail_transcribe(path: Path, **kwargs):  # pragma: no cover - ensure skip
        raise AssertionError("should not transcribe on resume")

    monkeypatch.setattr(audio_to_midi_batch, "_transcribe_stem", fail_transcribe)
    audio_to_midi_batch.main([str(song_dir), str(out_dir), "--resume"])


def test_tempo_written(tmp_path, monkeypatch):
    song_dir = tmp_path / "song"
    out_dir = tmp_path / "out"
    song_dir.mkdir()
    sr = 22050
    t = np.linspace(0, 1, sr, False)
    wave = 0.1 * np.sin(2 * np.pi * 440 * t)
    _write(song_dir / "a.wav", wave, sr)

    def tempo_stub(
        path: Path,
        *,
        step_size: int = 10,
        conf_threshold: float = 0.5,
        min_dur: float = 0.05,
        auto_tempo: bool = True,
    ) -> StemResult:
        inst = pretty_midi.Instrument(program=0, name=path.stem)
        inst.notes.append(
            pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=min_dur)
        )
        return StemResult(inst, 100.0)

    monkeypatch.setattr(audio_to_midi_batch, "_transcribe_stem", tempo_stub)
    audio_to_midi_batch.main([str(song_dir), str(out_dir), "--auto-tempo"])
    midi_path = out_dir / song_dir.name / "a.mid"
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    tempi, _ = pm.get_tempo_changes()
    assert tempi[0] == pytest.approx(100.0)


def test_auto_tempo_flag(tmp_path, monkeypatch):
    song_dir = tmp_path / "song"
    out_auto = tmp_path / "out_auto"
    out_off = tmp_path / "out_off"
    song_dir.mkdir()
    sr = 22050
    t = np.linspace(0, 1, sr, False)
    wave = 0.1 * np.sin(2 * np.pi * 440 * t)
    _write(song_dir / "a.wav", wave, sr)

    class FakeCrepe:
        @staticmethod
        def predict(audio, sr, step_size=10, model_capacity="full", verbose=0):
            time = np.array([0.0, 1.0])
            freq = np.array([440.0, 440.0])
            conf = np.array([1.0, 1.0])
            return time, freq, conf, None

    def fake_load(path, sr=16000, mono=True):
        return wave, sr

    beat_mod = types.SimpleNamespace(beat_track=lambda y, sr, trim=False: (150.0, None))
    fake_librosa = types.SimpleNamespace(load=fake_load, beat=beat_mod)

    monkeypatch.setattr(audio_to_midi_batch, "crepe", FakeCrepe)
    monkeypatch.setattr(audio_to_midi_batch, "librosa", fake_librosa)

    audio_to_midi_batch.main([str(song_dir), str(out_auto), "--auto-tempo"])
    pm = pretty_midi.PrettyMIDI(str(out_auto / song_dir.name / "a.mid"))
    tempi, _ = pm.get_tempo_changes()
    assert tempi[0] == pytest.approx(150.0)

    audio_to_midi_batch.main([str(song_dir), str(out_off), "--no-auto-tempo"])
    pm2 = pretty_midi.PrettyMIDI(str(out_off / song_dir.name / "a.mid"))
    tempi2, _ = pm2.get_tempo_changes()
    assert tempi2[0] == pytest.approx(120.0)


@pytest.mark.parametrize(
    "strategy,expected",
    [("first", 100.0), ("median", 125.0), ("ignore", 120.0)],
)
def test_tempo_strategy(tmp_path, monkeypatch, strategy, expected):
    song_dir = tmp_path / "song"
    out_dir = tmp_path / "out"
    song_dir.mkdir()
    sr = 22050
    t = np.linspace(0, 1, sr, False)
    wave = 0.1 * np.sin(2 * np.pi * 440 * t)
    _write(song_dir / "a.wav", wave, sr)
    _write(song_dir / "b.wav", wave, sr)

    tempos = [100.0, 150.0]

    def stub(
        path: Path,
        *,
        step_size: int = 10,
        conf_threshold: float = 0.5,
        min_dur: float = 0.05,
        auto_tempo: bool = True,
    ) -> StemResult:
        inst = pretty_midi.Instrument(program=0, name=path.stem)
        inst.notes.append(
            pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=min_dur)
        )
        tempo = tempos.pop(0)
        return StemResult(inst, tempo)

    monkeypatch.setattr(audio_to_midi_batch, "_transcribe_stem", stub)

    audio_to_midi_batch.main(
        [
            str(song_dir),
            str(out_dir),
            "--merge",
            "--tempo-strategy",
            strategy,
        ]
    )
    pm = pretty_midi.PrettyMIDI(str(out_dir / f"{song_dir.name}.mid"))
    tempi, _ = pm.get_tempo_changes()
    assert tempi[0] == pytest.approx(expected)
