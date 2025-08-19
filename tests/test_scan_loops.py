import os
from tempfile import NamedTemporaryFile
from typing import Any

import pytest
pytest.importorskip("mido")
pytest.importorskip("numpy")
import mido
import pretty_midi

import scripts.scan_loops as scan_loops
from scripts.scan_loops import _ensure_tempo
from utilities.pretty_midi_safe import pm_to_mido


def _make_tempo_less_pm() -> pretty_midi.PrettyMIDI:
    mid = mido.MidiFile()
    inst_track = mido.MidiTrack()
    meta_track = mido.MidiTrack()
    mid.tracks.extend([inst_track, meta_track])
    inst_track.append(mido.Message("note_on", note=60, velocity=100, time=0))
    inst_track.append(mido.Message("note_off", note=60, velocity=0, time=480))
    meta_track.append(mido.MetaMessage("track_name", name="meta", time=0))
    meta_track.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    with NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
        mid.save(tmp.name)
        path = tmp.name
    pm = pretty_midi.PrettyMIDI(path)
    os.remove(path)
    return pm


def test_ensure_tempo_injection() -> None:
    pm = _make_tempo_less_pm()
    pm2 = _ensure_tempo(pm, 120.0, "default")
    mid = pm_to_mido(pm2)
    track = mid.tracks[1]
    assert track[0].type == "set_tempo" and track[0].time == 0
    assert track[1].type == "track_name"
    assert track[2].type == "time_signature"
    assert getattr(pm2, "tempo_injected", False)
    assert pm2.tempo_source == "default"


def test_tempfile_cleanup_on_exception(tmp_path, monkeypatch) -> None:
    pm = _make_tempo_less_pm()
    tmp_file = tmp_path / "tmp.mid"

    class Dummy:
        def __enter__(self) -> Any:
            self.fh = tmp_file.open("wb")
            return self.fh

        def __exit__(self, exc_type, exc, tb) -> None:
            self.fh.close()

        @property
        def name(self) -> str:  # pragma: no cover - compatibility
            return str(tmp_file)

    monkeypatch.setattr(scan_loops, "NamedTemporaryFile", lambda suffix, delete=False: Dummy())

    def boom(_: str) -> pretty_midi.PrettyMIDI:
        raise RuntimeError("boom")

    monkeypatch.setattr(scan_loops.pretty_midi, "PrettyMIDI", boom)
    with pytest.raises(RuntimeError):
        scan_loops._ensure_tempo(pm, 120.0, "default")
    assert not tmp_file.exists()
