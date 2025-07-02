import logging
import time

from utilities.live_buffer import LiveBuffer


def slow_gen(idx: int) -> int:
    time.sleep(0.05)
    return idx


def test_live_buffer_integration(caplog):
    buf = LiveBuffer(slow_gen, buffer_ahead=2, parallel_bars=1, warn_level=logging.ERROR)
    caplog.set_level(logging.WARNING)
    results = []
    for _ in range(5):
        results.append(buf.get_next())
        time.sleep(0.1)
    buf.shutdown()
    assert results == list(range(5))
    assert not any("underrun" in r.message for r in caplog.records)
