import os
from tempfile import NamedTemporaryFile

import pytest

mido = pytest.importorskip("mido")
np = pytest.importorskip("numpy")
pretty_midi = pytest.importorskip("pretty_midi")

from utilities.audio_to_midi_batch import _set_initial_tempo
from utilities.groove_sampler_v2 import _ensure_tempo


def _tempo_less_pm() -> pretty_midi.PrettyMIDI:
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    track.append(mido.Message("note_on", note=60, velocity=100, time=0))
    track.append(mido.Message("note_off", note=60, velocity=0, time=480))
    mid.tracks.append(track)
    with NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
        mid.save(tmp.name)
        path = tmp.name
    try:
        return pretty_midi.PrettyMIDI(path)
    finally:
        os.remove(path)


def test_pretty_midi_compat_contract() -> None:
    pm = _tempo_less_pm()
    pm = _ensure_tempo(pm, 120.0)
    times, bpms = pm.get_tempo_changes()
    assert len(times) == len(bpms) > 0
    assert times[0] == pytest.approx(0.0, abs=1e-3)
    assert np.all(np.diff(times) >= -1e-9)
    assert np.all(np.asarray(bpms) > 0)


def test_initial_tempo_injected() -> None:
    pm = pretty_midi.PrettyMIDI()
    scale = 60.0 / (pm.resolution * 90.0)
    pm._tick_scales = [(pm.resolution, scale)]  # type: ignore[attr-defined]
    _set_initial_tempo(pm, 120.0)
    times, bpms = pm.get_tempo_changes()
    assert times[0] == pytest.approx(0.0, abs=1e-3)
    assert bpms[0] == pytest.approx(120.0, rel=1e-6, abs=1e-6)
    assert np.all(np.asarray(bpms) > 0)
