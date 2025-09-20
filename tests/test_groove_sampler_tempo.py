import io
from pathlib import Path

import pytest

pretty_midi = pytest.importorskip("pretty_midi")
mido = pytest.importorskip("mido")

from utilities import groove_sampler_v2 as module

_safe_read_bpm = module._safe_read_bpm
train = module.train


def make_pm(bpm: float | None = None, extra: list[float] | None = None):
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    if bpm is not None:
        track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0))
    if extra:
        for t in extra:
            track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(t), time=1))
    track.append(mido.Message("note_on", note=36, velocity=100, time=0))
    track.append(mido.Message("note_off", note=36, velocity=0, time=240))
    bio = io.BytesIO()
    mid.save(file=bio)
    bio.seek(0)
    pm = pretty_midi.PrettyMIDI(bio)
    return pm, mid


def test_safe_read_bpm_basic():
    pm, _ = make_pm(bpm=120)
    bpm = _safe_read_bpm(pm, default_bpm=100.0, fold_halves=False)
    assert bpm == pytest.approx(120.0)
    assert _safe_read_bpm.last_source == "pretty_midi"


def test_safe_read_bpm_default_120():
    pm, _ = make_pm(bpm=None)
    bpm = _safe_read_bpm(pm, default_bpm=120.0, fold_halves=False)
    assert bpm == pytest.approx(120.0)
    assert _safe_read_bpm.last_source == "default"


def test_safe_read_bpm_first_tempo():
    pm, _ = make_pm(bpm=110, extra=[150])
    bpm = _safe_read_bpm(pm, default_bpm=120.0, fold_halves=False)
    assert bpm == pytest.approx(110.0)


def test_fold_halves():
    pm, _ = make_pm(bpm=60)
    bpm = _safe_read_bpm(pm, default_bpm=120.0, fold_halves=True)
    assert bpm == pytest.approx(120.0)



def test_train_inject_default_tempo(tmp_path: Path):
    pm, _ = make_pm(bpm=None)
    midi_path = tmp_path / "no_tempo.mid"
    pm.write(str(midi_path))
    model = train(tmp_path, inject_default_tempo=120.0)
    assert model.files_scanned == 1
    assert model.total_events > 0
    pm2 = pretty_midi.PrettyMIDI(str(midi_path))
    tempi, _t = pm2.get_tempo_changes()
    assert len(tempi) == 0

