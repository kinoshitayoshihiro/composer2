import io
from pathlib import Path

import pytest

pretty_midi = pytest.importorskip("pretty_midi")
mido = pytest.importorskip("mido")

from utilities import groove_sampler_v2 as module

_safe_read_bpm = module._safe_read_bpm
train = module.train


def make_pm(bpm=None, extra=None):
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


def test_safe_read_bpm_default_and_skip(tmp_path: Path):
    pm, mid = make_pm(bpm=None)
    bpm = _safe_read_bpm(pm, default_bpm=130.0, fold_halves=False)
    assert bpm == pytest.approx(130.0)
    assert _safe_read_bpm.last_source == "default"
    midi_path = tmp_path / "no_tempo.mid"
    mid.save(midi_path)
    model = train(tmp_path, skip_no_tempo=True, default_bpm=130.0)
    assert model.files_skipped == 1


def test_safe_read_bpm_first_tempo():
    pm, _ = make_pm(bpm=110, extra=[150])
    bpm = _safe_read_bpm(pm, default_bpm=120.0, fold_halves=False)
    assert bpm == pytest.approx(110.0)


def test_fold_halves():
    pm, _ = make_pm(bpm=60)
    bpm = _safe_read_bpm(pm, default_bpm=120.0, fold_halves=True)
    assert bpm == pytest.approx(120.0)
