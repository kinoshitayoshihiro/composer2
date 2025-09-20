import math
from pathlib import Path

import pytest

mido = pytest.importorskip("mido")
pretty_midi = pytest.importorskip("pretty_midi")

from utilities.groove_sampler_v2 import train


def make_midi(
    path: Path,
    *,
    tempo: bool = True,
    bars: float = 1.0,
    notes: int = 8,
    channel: int = 0
):
    mid = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    if tempo:
        track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(120), time=0))
    ticks = 0
    for _ in range(notes):
        track.append(
            mido.Message("note_on", note=60, velocity=64, time=0, channel=channel)
        )
        track.append(
            mido.Message("note_off", note=60, velocity=64, time=60, channel=channel)
        )
        ticks += 60
    total_ticks = int(bars * 4 * mid.ticks_per_beat)
    track.append(mido.MetaMessage("end_of_track", time=max(0, total_ticks - ticks)))
    mid.save(path)


def test_min_bars_and_notes(tmp_path):
    make_midi(tmp_path / "ok.mid", bars=2.0, notes=8)
    make_midi(tmp_path / "short.mid", bars=0.5, notes=8)
    make_midi(tmp_path / "few.mid", bars=2.0, notes=4)
    model = train(tmp_path, n=1, min_bars=1.0, min_notes=8)
    assert model.files_scanned == 3
    assert model.files_skipped == 2
    assert model.file_weights[0] == pytest.approx(math.sqrt(2.0), rel=1e-3)


def test_drum_and_pitched_filters(tmp_path):
    make_midi(tmp_path / "d.mid", channel=9)
    make_midi(tmp_path / "p.mid", channel=0)
    m1 = train(tmp_path, n=1, drum_only=True)
    assert m1.files_skipped == 1
    m2 = train(tmp_path, n=1, pitched_only=True)
    assert m2.files_skipped == 1


def test_fill_exclusion(tmp_path):
    make_midi(tmp_path / "beat_fill.mid")
    model = train(tmp_path, n=1, exclude_fills=True)
    assert model.files_skipped == 1


def test_inject_default_tempo(tmp_path):
    path = tmp_path / "notempo.mid"
    make_midi(path, tempo=False)
    train(tmp_path, n=1, inject_default_tempo=100.0)
    pm = pretty_midi.PrettyMIDI(str(path))
    _times, tempi = pm.get_tempo_changes()
    assert pytest.approx(tempi[0]) == 100.0
