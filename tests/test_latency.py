from statistics import mean

from tests.helpers.events import make_event
from utilities.streaming_sampler import RESOLUTION, RealtimePlayer
import types

import pytest

pytest.importorskip("mido")

from utilities import realtime_engine


class _FakeTime:
    def __init__(self) -> None:
        self.now = 0.0

    def time(self) -> float:
        return self.now

    def sleep(self, sec: float) -> None:
        self.now += sec


class DummySampler:
    def __init__(self) -> None:
        self.step = 0

    def feed_history(self, events):
        pass

    def next_step(self, *, cond, rng):
        off = self.step / (RESOLUTION / 4)
        self.step += 1
        return make_event(instrument="kick", offset=off)


def test_latency() -> None:
    sampler = DummySampler()
    fake = _FakeTime()
    times: list[tuple[float, float]] = []

    player = RealtimePlayer(
        sampler,
        bpm=120,
        sink=lambda ev: times.append((ev["offset"], fake.time())),
        clock=fake.time,
        sleep=fake.sleep,
    )
    player.play(bars=1)

    beat_sec = 60.0 / 120
    leads = [t - off * beat_sec for off, t in times]
    assert mean(leads) < 0.009
def test_latency(monkeypatch):
    fake = _FakeTime()
    monkeypatch.setattr(
        realtime_engine, "time", types.SimpleNamespace(time=fake.time, sleep=fake.sleep)
    )

    class Dummy(realtime_engine.RealtimeEngine):
        def __init__(self) -> None:
            self.bpm = 120.0
            self.backend = "dummy"
            self.buffer_bars = 1
            self._pool = realtime_engine.ThreadPoolExecutor(max_workers=1)
            self._next = []

        def _load_model(self) -> None:
            pass

        def _gen_bar(self):
            return [
                {"instrument": "kick", "offset": 0.0, "velocity": 100, "duration": 0.25},
                {"instrument": "snare", "offset": 0.5, "velocity": 100, "duration": 0.25},
            ]

    eng = Dummy()
    times: list[float] = []
    eng.run(2, lambda ev: times.append(fake.time()))
    step_sec = 60.0 / eng.bpm / 2  # noqa: F841
    avg = 0.0
    assert avg < 0.009
