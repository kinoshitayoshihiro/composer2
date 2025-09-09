import types
import numpy as np

from utilities import loudness_meter as lm


class DummyStream:
    def __init__(self, *, channels, samplerate, callback, device=None):
        self.callback = callback

    def start(self):
        data = np.zeros((samplerate := 44100, 1), dtype=np.float32)
        self.callback(data, samplerate, None, None)

    def stop(self):
        pass

    def close(self):
        pass


def test_realtime_loudness_meter(monkeypatch):
    sd_stub = types.ModuleType("sd")
    sd_stub.InputStream = DummyStream
    monkeypatch.setattr(lm, "sd", sd_stub)
    meter = lm.RealtimeLoudnessMeter(sample_rate=44100)
    meter.start(None)
    val = meter.get_current_lufs()
    assert isinstance(val, float)
    meter.stop()
