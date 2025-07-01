from eval import latency
from utilities.streaming_sampler import RealtimePlayer


class DummySampler:
    def __init__(self) -> None:
        self.step = 0

    def feed_history(self, events: list[tuple[int, str]]) -> None:
        return

    def next_step(self, *, cond: dict | None, rng) -> dict:
        ev = {"instrument": "kick", "offset": self.step / 4}
        self.step = (self.step + 1) % 16
        return ev


def test_latency_measure() -> None:
    clock = 0.0

    def _clock() -> float:
        return clock

    def _sleep(d: float) -> None:
        nonlocal clock
        clock += d

    player = RealtimePlayer(DummySampler(), bpm=120.0, clock=_clock, sleep=_sleep)
    avg = latency.measure_latency(player, bars=1)
    assert avg < 10.0
