from __future__ import annotations

from .streaming_sampler import BaseSampler, RealtimePlayer


def play_live(sampler: BaseSampler, bpm: float) -> None:
    """Stream events from ``sampler`` in real time until interrupted."""
    player = RealtimePlayer(sampler, bpm=bpm)
    try:
        while True:
            player.play(bars=1)
    except KeyboardInterrupt:
        return
