from utilities.vibrato_engine import generate_vibrato


def test_generate_vibrato_waveform():
    events = generate_vibrato(0.2, 0.5, 5.0, step=0.05)
    bends = [e["value"] for e in events if e["type"] == "pitchwheel"]
    assert bends == [0, 2048, 0, -2048, 0]
    touches = [e["value"] for e in events if e["type"] == "aftertouch"]
    assert len(touches) == len(bends)
