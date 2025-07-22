import os
import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("COMPOSER_CI_STUBS") != "1",
    reason="requires COMPOSER_CI_STUBS=1",
)


def test_music21_stub():
    import music21

    for name in [
        "pitch",
        "note",
        "stream",
        "converter",
        "instrument",
        "articulations",
        "harmony",
        "key",
        "meter",
        "interval",
    ]:
        assert hasattr(music21, name)

    from music21 import (
        pitch,
        note,
        stream,
        converter,
        instrument,
        articulations,
        harmony,
        key,
        meter,
        interval,
    )

    assert hasattr(pitch, "Pitch")
    assert hasattr(note, "Note")
    assert hasattr(stream, "Stream")
    assert hasattr(converter, "Converter")
    assert hasattr(instrument, "Instrument")
    assert hasattr(articulations, "Articulation")
    assert hasattr(harmony, "ChordSymbol")
    assert hasattr(key, "Key")
    assert hasattr(meter, "TimeSignature")
    assert hasattr(interval, "Interval")


def test_scipy_signal_stub():
    from scipy import signal

    assert callable(signal.hamming)
    assert callable(signal.butter)
    assert callable(signal.lfilter)
