import importlib.util
import sys
import types
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pretty_midi = pytest.importorskip("pretty_midi")
import wave
import multiprocessing
import logging
from utilities import pb_math

np = pytest.importorskip("numpy")

spec = importlib.util.spec_from_file_location(
    "utilities.audio_to_midi_batch",
    Path(__file__).resolve().parents[1] / "utilities" / "audio_to_midi_batch.py",
)
audio_to_midi_batch = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = audio_to_midi_batch
spec.loader.exec_module(audio_to_midi_batch)
StemResult = audio_to_midi_batch.StemResult


def _stub_transcribe(
    path: Path,
    *,
    step_size: int = 10,
    conf_threshold: float = 0.5,
    min_dur: float = 0.05,
    auto_tempo: bool = True,
    enable_bend: bool = True,
    bend_range_semitones: float = 2.0,
    bend_alpha: float = 0.25,
    bend_fixed_base: bool = False,
    cc_strategy: str = "none",
    cc11_smoothing_ms: int = 80,
    sustain_threshold: float = 0.6,
    cc11_min_dt_ms: int = 30,
    cc11_min_delta: int = 3,
    **kwargs,
) -> StemResult:
    inst = pretty_midi.Instrument(program=0, name=path.stem)
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=min_dur))
    return types.SimpleNamespace(instrument=inst, tempo=120.0)


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


def test_rpn_emitted(tmp_path, monkeypatch):
    song_dir = tmp_path / "song"
    out_dir = tmp_path / "out"
    song_dir.mkdir()
    sr = 22050
    t = np.linspace(0, 1, sr, False)
    wave = 0.1 * np.sin(2 * np.pi * 440 * t)
    _write(song_dir / "a.wav", wave, sr)

    monkeypatch.setattr(audio_to_midi_batch, "_transcribe_stem", _stub_transcribe)

    audio_to_midi_batch.main([str(song_dir), str(out_dir), "--bend-range-semitones", "12.34"])
    midi_path = out_dir / song_dir.name / "a.mid"
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    ccs = pm.instruments[0].control_changes
    cc_map = {(cc.number, cc.value) for cc in ccs}
    assert (101, 0) in cc_map
    assert (100, 0) in cc_map
    assert (6, 12) in cc_map
    assert (38, 34) in cc_map


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
        enable_bend: bool = True,
        bend_range_semitones: float = 2.0,
        bend_alpha: float = 0.25,
        bend_fixed_base: bool = False,
        **kwargs,
    ) -> StemResult:
        inst = pretty_midi.Instrument(program=0, name=path.stem)
        inst.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=min_dur))
        return types.SimpleNamespace(instrument=inst, tempo=100.0)

    monkeypatch.setattr(audio_to_midi_batch, "_transcribe_stem", tempo_stub)
    audio_to_midi_batch.main([str(song_dir), str(out_dir), "--auto-tempo"])
    midi_path = out_dir / song_dir.name / "a.mid"
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    _, tempi = pm.get_tempo_changes()
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
    _, tempi = pm.get_tempo_changes()
    assert tempi[0] == pytest.approx(150.0)

    audio_to_midi_batch.main([str(song_dir), str(out_off), "--no-auto-tempo"])
    pm2 = pretty_midi.PrettyMIDI(str(out_off / song_dir.name / "a.mid"))
    _, tempi2 = pm2.get_tempo_changes()
    assert tempi2[0] == pytest.approx(120.0)


def test_pitch_bend_generation(tmp_path, monkeypatch):
    path = tmp_path / "vib.wav"
    sr = 16000
    t = np.linspace(0, 1, sr, False)
    wave = np.zeros_like(t)
    _write(path, wave, sr)

    times = np.arange(0, 1, 0.01)
    freq = 440 * 2 ** (np.sin(2 * np.pi * 5 * times) / 12)
    conf = np.ones_like(times)

    class FakeCrepe:
        @staticmethod
        def predict(audio, sr, step_size=10, model_capacity="full", verbose=0):
            return times, freq, conf, None

    def fake_load(p, sr=16000, mono=True):
        return wave, sr

    monkeypatch.setattr(audio_to_midi_batch, "crepe", FakeCrepe)
    monkeypatch.setattr(audio_to_midi_batch, "librosa", types.SimpleNamespace(load=fake_load))

    res = audio_to_midi_batch._transcribe_stem(
        path,
        enable_bend=True,
        bend_range_semitones=2.0,
        bend_alpha=0.5,
        auto_tempo=False,
    )
    pb = res.instrument.pitch_bends
    assert pb
    assert pb[0].pitch == 0
    assert pb[-1].pitch == 0
    bends = [p.pitch for p in pb]
    max_b, min_b = max(bends), min(bends)

    nn = pretty_midi.hz_to_note_number(freq)
    devs = nn - np.round(nn)
    alpha = 0.5
    ema = 0.0
    emas = []
    for d in devs:
        ema = alpha * d + (1 - alpha) * ema
        emas.append(ema)
    expected = int(round(np.max(np.abs(emas)) / 2.0 * pb_math.PB_MAX))
    assert max_b == pytest.approx(expected, rel=0.1)
    assert min_b == pytest.approx(-expected, rel=0.1)


def test_pitch_bend_disabled(tmp_path, monkeypatch):
    path = tmp_path / "vib.wav"
    sr = 16000
    t = np.linspace(0, 1, sr, False)
    wave = np.zeros_like(t)
    _write(path, wave, sr)

    times = np.arange(0, 1, 0.01)
    freq = 440 * np.ones_like(times)
    conf = np.ones_like(times)

    class FakeCrepe:
        @staticmethod
        def predict(audio, sr, step_size=10, model_capacity="full", verbose=0):
            return times, freq, conf, None

    def fake_load(p, sr=16000, mono=True):
        return wave, sr

    monkeypatch.setattr(audio_to_midi_batch, "crepe", FakeCrepe)
    monkeypatch.setattr(audio_to_midi_batch, "librosa", types.SimpleNamespace(load=fake_load))

    res = audio_to_midi_batch._transcribe_stem(
        path,
        enable_bend=False,
        auto_tempo=False,
    )
    assert res.instrument.pitch_bends == []


def test_pitch_bend_fallback_no_crepe(tmp_path, monkeypatch, caplog):
    path = tmp_path / "a.wav"
    sr = 16000
    t = np.linspace(0, 1, sr, False)
    wave = np.zeros_like(t)
    _write(path, wave, sr)

    monkeypatch.setattr(audio_to_midi_batch, "crepe", None)
    monkeypatch.setattr(audio_to_midi_batch, "librosa", None)

    def fb(path: Path, *, min_dur: float, tempo: float | None = None):
        inst = pretty_midi.Instrument(program=0, name=path.stem)
        return StemResult(inst, tempo)

    monkeypatch.setattr(audio_to_midi_batch, "_fallback_transcribe_stem", fb)

    with caplog.at_level(logging.INFO):
        res = audio_to_midi_batch._transcribe_stem(path, enable_bend=True, auto_tempo=False)
    assert res.instrument.pitch_bends == []
    assert any("Pitch-bend disabled" in r.message for r in caplog.records)


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
        enable_bend: bool = True,
        bend_range_semitones: float = 2.0,
        bend_alpha: float = 0.25,
        bend_fixed_base: bool = False,
        **kwargs,
    ) -> StemResult:
        inst = pretty_midi.Instrument(program=0, name=path.stem)
        inst.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=min_dur))
        tempo = tempos.pop(0)
        return types.SimpleNamespace(instrument=inst, tempo=tempo)

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
    _, tempi = pm.get_tempo_changes()
    assert tempi[0] == pytest.approx(expected)


def test_cc11_energy(tmp_path, monkeypatch):
    in_dir = tmp_path / "song"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    sr = 22050
    t = np.linspace(0, 2, sr * 2, False)
    audio = np.concatenate([np.ones(sr), np.linspace(1.0, 0.0, sr)])
    _write(in_dir / "voice.wav", audio, sr)

    def stub(path: Path, **kwargs):
        inst = pretty_midi.Instrument(program=0, name=path.stem)
        inst.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=2.0))
        events = audio_to_midi_batch.cc_utils.energy_to_cc11(
            audio,
            sr,
            smooth_ms=kwargs.get("cc11_smoothing_ms", 80),
            strategy=kwargs.get("cc_strategy", "none"),
        )
        prev = -1
        for t, v in events:
            if v != prev:
                inst.control_changes.append(
                    pretty_midi.ControlChange(number=11, value=v, time=float(t))
                )
                prev = v
        return types.SimpleNamespace(instrument=inst, tempo=120.0)

    monkeypatch.setattr(audio_to_midi_batch, "_transcribe_stem", stub)
    audio_to_midi_batch.main([str(in_dir), str(out_dir), "--cc-strategy", "energy"])
    pm = pretty_midi.PrettyMIDI(str(out_dir / in_dir.name / "voice.mid"))
    vals = [cc.value for cc in pm.instruments[0].control_changes if cc.number == 11]
    assert sum(vals[-5:]) < sum(vals[:5])


def test_sustain_threshold(tmp_path, monkeypatch):
    in_dir = tmp_path / "piano_song"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    sr = 22050
    audio = np.ones(sr)
    _write(in_dir / "piano.wav", audio, sr)

    def stub(path: Path, **kwargs):
        inst = pretty_midi.Instrument(program=0, name=path.stem)
        inst.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=0.4))
        inst.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.6, end=1.0))
        if kwargs.get("cc_strategy", "none") != "none":
            events = audio_to_midi_batch.cc_utils.energy_to_cc11(
                audio,
                sr,
                smooth_ms=kwargs.get("cc11_smoothing_ms", 80),
                strategy=kwargs.get("cc_strategy", "none"),
            )
            prev = -1
            for t, v in events:
                if v != prev:
                    inst.control_changes.append(
                        pretty_midi.ControlChange(number=11, value=v, time=float(t))
                    )
                    prev = v
        for t, v in audio_to_midi_batch.cc_utils.infer_cc64_from_overlaps(
            inst.notes, kwargs.get("sustain_threshold", 0.5)
        ):
            inst.control_changes.append(
                pretty_midi.ControlChange(number=64, value=v, time=float(t))
            )
        return types.SimpleNamespace(instrument=inst, tempo=None)

    monkeypatch.setattr(audio_to_midi_batch, "_transcribe_stem", stub)
    audio_to_midi_batch.main(
        [
            str(in_dir),
            str(out_dir),
            "--sustain-threshold",
            "0.5",
        ]
    )
    pm = pretty_midi.PrettyMIDI(str(out_dir / in_dir.name / "piano.mid"))
    vals = [cc.value for cc in pm.instruments[0].control_changes if cc.number == 64]
    assert 127 in vals and 0 in vals


def test_cc11_sparsify(tmp_path):
    sr = 22050
    audio = np.ones(sr * 2)
    inst = pretty_midi.Instrument(program=0)
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=2.0))
    events = audio_to_midi_batch.cc_utils.energy_to_cc11(audio, sr, smooth_ms=0, strategy="energy")
    prev = -1
    for t, v in events:
        if v != prev:
            inst.control_changes.append(
                pretty_midi.ControlChange(number=11, value=v, time=float(t))
            )
            prev = v
    events = [cc for cc in inst.control_changes if cc.number == 11]
    assert len(events) <= 70


def test_tempo_lock_anchor_fold_halves(tmp_path, monkeypatch):
    song_dir = tmp_path / "song"
    out_dir = tmp_path / "out"
    song_dir.mkdir()
    sr = 22050
    t = np.linspace(0, 1, sr, False)
    wave = 0.1 * np.sin(2 * np.pi * 440 * t)
    _write(song_dir / "drum.wav", wave, sr)
    _write(song_dir / "bass.wav", wave, sr)

    def stub(path: Path, **_):
        inst = pretty_midi.Instrument(program=0, name=path.stem)
        inst.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=0.1))
        tempo = 110.3 if "drum" in path.stem else 55.1
        return types.SimpleNamespace(instrument=inst, tempo=tempo)

    monkeypatch.setattr(audio_to_midi_batch, "_transcribe_stem", stub)

    audio_to_midi_batch.main(
        [
            str(song_dir),
            str(out_dir),
            "--tempo-lock",
            "anchor",
            "--tempo-anchor-pattern",
            "(?i)drum",
            "--tempo-fold-halves",
        ]
    )
    midi_dir = out_dir / song_dir.name
    tempos = set()
    for p in midi_dir.glob("*.mid"):
        _, tempi = pretty_midi.PrettyMIDI(str(p)).get_tempo_changes()
        tempos.add(round(float(tempi[0]), 1))
    assert tempos == {round(110.3, 1)}


def test_tempo_lock_median_fold_halves(tmp_path, monkeypatch):
    song_dir = tmp_path / "song"
    out_dir = tmp_path / "out"
    song_dir.mkdir()
    sr = 22050
    t = np.linspace(0, 1, sr, False)
    wave = 0.1 * np.sin(2 * np.pi * 440 * t)
    _write(song_dir / "a.wav", wave, sr)
    _write(song_dir / "b.wav", wave, sr)
    _write(song_dir / "c.wav", wave, sr)

    tempos = {"a": 110.3, "b": 55.1, "c": 220.6}

    def stub(path: Path, **_):
        inst = pretty_midi.Instrument(program=0, name=path.stem)
        inst.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=0.1))
        tempo = tempos[path.stem]
        return types.SimpleNamespace(instrument=inst, tempo=tempo)

    monkeypatch.setattr(audio_to_midi_batch, "_transcribe_stem", stub)

    audio_to_midi_batch.main(
        [
            str(song_dir),
            str(out_dir),
            "--tempo-lock",
            "median",
            "--tempo-fold-halves",
        ]
    )
    midi_dir = out_dir / song_dir.name
    tempos_out = set()
    for p in midi_dir.glob("*.mid"):
        _, tempi = pretty_midi.PrettyMIDI(str(p)).get_tempo_changes()
        tempos_out.add(round(float(tempi[0]), 1))
    assert tempos_out == {round(110.3, 1)}


def test_tempo_lock_value(tmp_path, monkeypatch):
    song_dir = tmp_path / "song"
    out_dir = tmp_path / "out"
    song_dir.mkdir()
    sr = 22050
    t = np.linspace(0, 1, sr, False)
    wave = 0.1 * np.sin(2 * np.pi * 440 * t)
    _write(song_dir / "a.wav", wave, sr)
    _write(song_dir / "b.wav", wave, sr)

    def stub(path: Path, **_):
        inst = pretty_midi.Instrument(program=0, name=path.stem)
        inst.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=0.1))
        tempo = 90.0 if path.stem == "a" else 150.0
        return types.SimpleNamespace(instrument=inst, tempo=tempo)

    monkeypatch.setattr(audio_to_midi_batch, "_transcribe_stem", stub)

    audio_to_midi_batch.main(
        [
            str(song_dir),
            str(out_dir),
            "--tempo-lock",
            "value",
            "--tempo-lock-value",
            "128",
        ]
    )
    midi_dir = out_dir / song_dir.name
    tempos = set()
    for p in midi_dir.glob("*.mid"):
        _, tempi = pretty_midi.PrettyMIDI(str(p)).get_tempo_changes()
        tempos.add(int(round(float(tempi[0]))))
    assert tempos == {128}


def test_tempo_lock_none_default(tmp_path, monkeypatch):
    song_dir = tmp_path / "song"
    out_dir = tmp_path / "out"
    song_dir.mkdir()
    sr = 22050
    t = np.linspace(0, 1, sr, False)
    wave = 0.1 * np.sin(2 * np.pi * 440 * t)
    _write(song_dir / "a.wav", wave, sr)
    _write(song_dir / "b.wav", wave, sr)

    tempos = {"a": 100.0, "b": 150.0}

    def stub(path: Path, **_):
        inst = pretty_midi.Instrument(program=0, name=path.stem)
        inst.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=0.1))
        tempo = tempos[path.stem]
        return types.SimpleNamespace(instrument=inst, tempo=tempo)

    monkeypatch.setattr(audio_to_midi_batch, "_transcribe_stem", stub)

    audio_to_midi_batch.main([str(song_dir), str(out_dir)])
    midi_dir = out_dir / song_dir.name
    tempos_out = set()
    for p in midi_dir.glob("*.mid"):
        _, tempi = pretty_midi.PrettyMIDI(str(p)).get_tempo_changes()
        tempos_out.add(int(round(float(tempi[0]))))
    assert tempos_out == {100, 150}


def test_tempo_lock_merge_anchor(tmp_path, monkeypatch):
    song_dir = tmp_path / "song"
    out_dir = tmp_path / "out"
    song_dir.mkdir()
    sr = 22050
    t = np.linspace(0, 1, sr, False)
    wave = 0.1 * np.sin(2 * np.pi * 440 * t)
    _write(song_dir / "drum.wav", wave, sr)
    _write(song_dir / "bass.wav", wave, sr)

    def stub(path: Path, **_):
        inst = pretty_midi.Instrument(program=0, name=path.stem)
        inst.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=0.1))
        tempo = 110.3 if "drum" in path.stem else 55.1
        return types.SimpleNamespace(instrument=inst, tempo=tempo)

    monkeypatch.setattr(audio_to_midi_batch, "_transcribe_stem", stub)

    audio_to_midi_batch.main(
        [
            str(song_dir),
            str(out_dir),
            "--merge",
            "--tempo-lock",
            "anchor",
            "--tempo-anchor-pattern",
            "(?i)drum",
            "--tempo-fold-halves",
        ]
    )
    midi_path = out_dir / f"{song_dir.name}.mid"
    _, tempi = pretty_midi.PrettyMIDI(str(midi_path)).get_tempo_changes()
    assert round(float(tempi[0]), 1) == round(110.3, 1)
