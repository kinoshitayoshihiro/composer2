from utilities.peak_synchroniser import PeakSynchroniser
from tests.helpers.events import make_event


def test_basic_alignment() -> None:
    peaks = [0.0, 0.5, 1.0]
    events = PeakSynchroniser.sync_events(peaks, [], tempo_bpm=120.0)
    ks = [
        (e["instrument"], round(e["offset"], 2))
        for e in events
        if e["instrument"] in {"kick", "snare"}
    ]
    assert ks[0][0] == "kick" and abs(ks[0][1] - 0.0) <= 0.03
    assert ks[1][0] == "snare" and abs(ks[1][1] - 1.0) <= 0.03
    assert ks[2][0] == "kick" and abs(ks[2][1] - 2.0) <= 0.03
    offs = [e["offset"] for e in events if e["instrument"] in {"kick", "snare"}]
    assert len(offs) == len(set(offs))


def test_pre_hit_lag() -> None:
    peaks = [0.0]
    events = PeakSynchroniser.sync_events(peaks, [], tempo_bpm=120.0, lag_ms=-20)
    offset_sec = events[0]["offset"] * 60.0 / 120.0
    assert -0.023 < offset_sec < -0.017


def test_clip_at_zero() -> None:
    events = PeakSynchroniser.sync_events(
        [0.0], [], tempo_bpm=120.0, lag_ms=-30, clip_at_zero=True
    )
    assert events[0]["offset"] == 0.0


def test_priority_replacement() -> None:
    base = [make_event(instrument="snare", offset=0.0)]
    events = PeakSynchroniser.sync_events([0.0], base, tempo_bpm=120.0)
    kicks = [e for e in events if e["instrument"] == "kick"]
    assert len(kicks) == 1
    assert 0.018 < kicks[0]["offset"] < 0.022
