"""Stub implementation of the Piano Companion plugin."""
from __future__ import annotations

from typing import Any, Dict

DEFAULT_PARAMETERS: Dict[str, Any] = {
    "TonePreset": "Default",
    "Intensity": 1.0,
    "Temperature": 0.7,
}


def get_default_parameters() -> Dict[str, Any]:
    """Return default parameter values."""
    return DEFAULT_PARAMETERS.copy()


def live_render(port: str, *, bpm: float = 120.0, buffer_bars: int = 2) -> None:
    """Render a dummy stream via :class:`RtMidiStreamer`."""
    from realtime import rtmidi_streamer

    class _DummyGen:
        def step(self, _ctx: list[dict[str, Any]]):
            return [{"pitch": 60, "velocity": 100, "offset": 0.0, "duration": 1.0}]

    streamer = rtmidi_streamer.RtMidiStreamer(port, _DummyGen())
    streamer.start(bpm=bpm, buffer_bars=buffer_bars, callback=None)
    try:
        streamer.on_tick()
    finally:
        streamer.stop()
