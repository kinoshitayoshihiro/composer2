import io
from pathlib import Path

import pytest

pretty_midi = pytest.importorskip("pretty_midi")
mido = pytest.importorskip("mido")

from utilities import groove_sampler_v2 as module  # noqa: E402

train = module.train
_resolve_tempo = module._resolve_tempo


def make_midi(bpm=None):
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    if bpm is not None:
        track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0))
    track.append(mido.Message("note_on", note=60, velocity=100, time=0))
    track.append(mido.Message("note_off", note=60, velocity=0, time=240))
    bio = io.BytesIO()
    mid.save(file=bio)
    bio.seek(0)
    pm = pretty_midi.PrettyMIDI(bio)
    return pm, mid


def test_policy_skip(tmp_path: Path):
    _, mid = make_midi(bpm=None)
    mid.save(tmp_path / "skip.mid")
    model = train(tmp_path, tempo_policy="skip")
    assert model.files_skipped == 1


def test_policy_fallback(tmp_path: Path):
    _, mid = make_midi(bpm=None)
    mid.save(tmp_path / "fb.mid")
    model = train(tmp_path, tempo_policy="fallback", fallback_bpm=100.0)
    assert model.files_skipped == 0


def test_policy_accept_warn(tmp_path: Path, caplog):
    _, mid = make_midi(bpm=None)
    mid.save(tmp_path / "acc.mid")
    with caplog.at_level("WARNING"):
        model = train(tmp_path, tempo_policy="accept_warn")
    assert model.files_skipped == 0
    assert any("Invalid tempo" in r.message for r in caplog.records)


def test_fold_halves():
    pm, _ = make_midi(bpm=55)
    bpm, reason = _resolve_tempo(
        pm,
        tempo_policy="accept",
        fallback_bpm=120.0,
        min_bpm=80.0,
        max_bpm=160.0,
        fold_halves=True,
    )
    assert bpm == pytest.approx(110.0)
    assert reason == "fold"


def test_tempo_verbose(tmp_path: Path, caplog):
    # one valid, one missing
    _, mid1 = make_midi(bpm=120)
    mid1.save(tmp_path / "a.mid")
    _, mid2 = make_midi(bpm=None)
    mid2.save(tmp_path / "b.mid")
    with caplog.at_level("INFO"):
        train(
            tmp_path,
            tempo_policy="skip",
            tempo_verbose=True,
        )
    assert any("tempo summary" in r.message for r in caplog.records)
    assert any("skipped files" in r.message for r in caplog.records)
