from statistics import mean

from tests.helpers.events import make_event
from utilities.streaming_sampler import RESOLUTION, RealtimePlayer


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
