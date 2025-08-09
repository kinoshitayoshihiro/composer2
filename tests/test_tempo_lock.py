import importlib.util
import sys
import types
from pathlib import Path

import pytest
pretty_midi = pytest.importorskip("pretty_midi")

spec = importlib.util.spec_from_file_location(
    "utilities.audio_to_midi_batch",
    Path(__file__).resolve().parents[1] / "utilities" / "audio_to_midi_batch.py",
)
audio_to_midi_batch = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = audio_to_midi_batch
spec.loader.exec_module(audio_to_midi_batch)


def test_fold_to_ref_boundaries():
    f = audio_to_midi_batch._fold_to_ref
    ref = 100.0
    assert f(74.9, ref) == pytest.approx(149.8)
    assert f(75.0, ref) == pytest.approx(75.0)
    assert f(75.1, ref) == pytest.approx(75.1)
    assert f(149.9, ref) == pytest.approx(149.9)
    assert f(150.0, ref) == pytest.approx(150.0)
    assert f(150.1, ref) == pytest.approx(75.05)


def test_invalid_anchor_pattern_fallback(tmp_path, monkeypatch):
    song_dir = tmp_path / "song"
    out_dir = tmp_path / "out"
    song_dir.mkdir()
    (song_dir / "a.wav").touch()
    (song_dir / "b.wav").touch()

    def stub(path: Path, **_):
        inst = pretty_midi.Instrument(program=0, name=path.stem)
        inst.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=0.1))
        tempo = 100.0 if path.stem == "a" else 150.0
        return types.SimpleNamespace(instrument=inst, tempo=tempo)

    monkeypatch.setattr(audio_to_midi_batch, "_transcribe_stem", stub)

    audio_to_midi_batch.main(
        [
            str(song_dir),
            str(out_dir),
            "--tempo-lock",
            "anchor",
            "--tempo-anchor-pattern",
            "[",
        ]
    )
    midi_dir = out_dir / song_dir.name
    tempos = set()
    for p in midi_dir.glob("*.mid"):
        _, tempi = pretty_midi.PrettyMIDI(str(p)).get_tempo_changes()
        tempos.add(round(float(tempi[0]), 1))
    assert tempos == {125.0}


def test_invalid_anchor_pattern_error(tmp_path, monkeypatch):
    song_dir = tmp_path / "song"
    out_dir = tmp_path / "out"
    song_dir.mkdir()
    (song_dir / "a.wav").touch()

    def stub(path: Path, **_):
        inst = pretty_midi.Instrument(program=0, name=path.stem)
        inst.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=0.1))
        return types.SimpleNamespace(instrument=inst, tempo=120.0)

    monkeypatch.setattr(audio_to_midi_batch, "_transcribe_stem", stub)

    with pytest.raises(SystemExit) as exc:
        audio_to_midi_batch.main(
            [
                str(song_dir),
                str(out_dir),
                "--tempo-lock",
                "anchor",
                "--tempo-anchor-pattern",
                "[",
                "--tempo-lock-fallback",
                "none",
            ]
        )
    assert exc.value.code == 2


def test_value_mode_requires_value(tmp_path, monkeypatch):
    song_dir = tmp_path / "song"
    out_dir = tmp_path / "out"
    song_dir.mkdir()
    (song_dir / "a.wav").touch()

    def stub(path: Path, **_):
        inst = pretty_midi.Instrument(program=0, name=path.stem)
        inst.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=0.1))
        return types.SimpleNamespace(instrument=inst, tempo=120.0)

    monkeypatch.setattr(audio_to_midi_batch, "_transcribe_stem", stub)

    with pytest.raises(SystemExit) as exc:
        audio_to_midi_batch.main(
            [str(song_dir), str(out_dir), "--tempo-lock", "value"]
        )
    assert exc.value.code == 2


def test_fallback_transcribe_without_numpy(tmp_path, monkeypatch):
    pretty_midi  # ensure dependency is loaded for skip logic
    dummy_wav = tmp_path / "x.wav"
    dummy_wav.touch()

    # Remove numpy and provide a fake scipy wavfile.read
    monkeypatch.setattr(audio_to_midi_batch, "np", None)
    fake_wavfile = types.SimpleNamespace(read=lambda p: (100, [0.0, 1.0, 0.0, 0.0, 1.0]))
    fake_io = types.SimpleNamespace(wavfile=fake_wavfile)
    sys.modules["scipy"] = types.SimpleNamespace(io=fake_io)
    sys.modules["scipy.io"] = fake_io
    sys.modules["scipy.io.wavfile"] = fake_wavfile

    res = audio_to_midi_batch._fallback_transcribe_stem(
        dummy_wav, min_dur=0.05, tempo=120.0
    )
    assert isinstance(res.instrument, pretty_midi.Instrument)
