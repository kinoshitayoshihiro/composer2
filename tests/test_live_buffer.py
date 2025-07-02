import logging
import time

from utilities.live_buffer import LiveBuffer


def slow_gen(idx: int):
    time.sleep(0.1)
    return idx


def test_live_buffer_warn(caplog):
    buf = LiveBuffer(slow_gen, buffer_ahead=1, parallel_bars=1)
    caplog.set_level(logging.WARNING)
    # force underrun by clearing the queue
    buf.buffer.clear()
    val = buf.get_next()
    buf.shutdown()
    assert val == 1
    assert any("underrun" in r.message for r in caplog.records)
