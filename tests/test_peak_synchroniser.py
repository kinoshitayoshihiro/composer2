from tools.peak_synchroniser import PeakSynchroniser
import pytest


def test_sync_moves_event_to_peak():
    peaks = [0.50]  # one peak at 0.5s
    events = [{"instrument": "kick", "offset": 0.0, "duration": 1.0}]
    synced = PeakSynchroniser.sync_events(
        peaks,
        events,
        tempo_bpm=120,
        lag_ms=10,
        min_distance_beats=0.25,
        sustain_threshold_ms=100,
    )
    # expect a new hit aligned to the peak around beat 1
    sec_per_beat = 60 / 120
    beat = 0.5 / sec_per_beat
    quant = round(beat * 2) / 2
    target = quant + (10 / 1000) / sec_per_beat
    target = round(target * 480) / 480
    hits = [
        e
        for e in synced
        if e["instrument"] in {"kick", "snare"} and abs(e["offset"] - target) < 0.05
    ]
    assert hits and hits[0]["offset"] == pytest.approx(target, abs=1e-3)
