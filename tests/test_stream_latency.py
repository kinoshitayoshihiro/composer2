import pytest

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
        self.feed_calls = 0

    def feed_history(self, events):
        if events:
            self.feed_calls += 1

    def next_step(self, *, cond, rng):
        off = self.step / (RESOLUTION / 4)
        self.step += 1
        return {"instrument": "kick", "offset": off, "duration": 0.25, "velocity": 100}

def test_latency() -> None:
    sampler = DummySampler()
    fake = _FakeTime()
    times: list[float] = []
    player = RealtimePlayer(
        sampler,
        bpm=120,
        sink=lambda ev: times.append(fake.time()),
        clock=fake.time,
        sleep=fake.sleep,
    )
    player.play(bars=1)

    assert sampler.feed_calls == 1
    assert len(times) >= 2
    step_sec = 60.0 / 120 / (RESOLUTION / 4)
    assert times[1] - times[0] == pytest.approx(step_sec, rel=0.01)
