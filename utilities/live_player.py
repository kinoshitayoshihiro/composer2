from __future__ import annotations

import time
from collections.abc import Iterable
from typing import Any, cast

from .streaming_sampler import BaseSampler, RealtimePlayer

try:  # optional dependency for MIDI output
    import mido
except Exception as e:  # pragma: no cover - optional dependency
    mido = None  # type: ignore
    _MIDO_ERROR = e
else:
    _MIDO_ERROR = None

Event = dict[str, Any]


def _play_sampler(sampler: BaseSampler, bpm: float) -> None:
    player = RealtimePlayer(sampler, bpm=bpm)
    try:
        while True:
            player.play(bars=1)
    except KeyboardInterrupt:
        return


def _play_events(events: Iterable[Event], bpm: float, port: str | None) -> None:
    if mido is None:
        raise ImportError("mido is required for live playback") from _MIDO_ERROR
    if port is None:
        names = mido.get_output_names()
        if not names:
            raise RuntimeError("No MIDI output ports")
        port = names[0]
    beat_sec = 60.0 / bpm
    start = time.time()
    with mido.open_output(port) as out:
        for ev in events:
            t = start + float(ev.get("offset", 0)) * beat_sec
            delay = t - time.time()
            if delay > 0:
                time.sleep(delay)
            note = 36 if ev.get("instrument") == "kick" else 38
            velocity = int(ev.get("velocity", 100))
            out.send(mido.Message("note_on", note=note, velocity=velocity))
            out.send(mido.Message("note_off", note=note, velocity=0))


def play_live(
    source: BaseSampler | Iterable[Event], bpm: float, port: str | None = None
) -> None:
    """Stream events live from ``source`` until interrupted."""
    if hasattr(source, "next_step"):
        _play_sampler(cast(BaseSampler, source), bpm)
    else:
        _play_events(cast(Iterable[Event], source), bpm, port)
