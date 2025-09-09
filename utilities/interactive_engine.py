from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

try:
    import mido
except Exception as e:  # pragma: no cover - optional
    mido = None  # type: ignore
    _MIDO_ERROR = e
else:
    _MIDO_ERROR = None

try:
    import transformers  # noqa: F401
except Exception:  # pragma: no cover - optional
    try:
        import pytest
    except Exception:
        pass
    else:
        pytest.importorskip("transformers")

from .ai_sampler import TransformerBassGenerator
from .bass_transformer import BassTransformer
from .live_buffer import LiveBuffer


class InteractiveEngine:
    """Trigger bass generation on incoming MIDI events."""

    def __init__(
        self,
        *,
        backend: str = "transformer",
        model_name: str = "gpt2-medium",
        bpm: float = 120.0,
        buffer_ms: int = 100,
        lookahead_bars: float = 0.5,
    ) -> None:
        if backend != "transformer":
            raise ValueError("Only transformer backend supported")
        self.generator = TransformerBassGenerator(model_name)
        self.bpm = bpm
        self.buffer_ms = buffer_ms
        self.lookahead_bars = lookahead_bars
        self.callbacks: list[Callable[[dict], None]] = []

    def add_callback(self, callback: Callable[[dict], None]) -> None:
        self.callbacks.append(callback)

    def _trigger(self, msg: Any) -> list[dict]:
        """Return generated events for ``msg`` if it is a note on."""
        if msg.type == "note_on" and msg.velocity > 0:
            events = self.generator.generate(
                [{"pitch": msg.note, "velocity": msg.velocity}],
                max(1, int(self.lookahead_bars)),
            )
            for cb in self.callbacks:
                for ev in events:
                    cb(ev)
            return events
        return []

    async def start(self, midi_in: str, midi_out: str) -> None:
        """Begin processing ``midi_in`` and sending to ``midi_out``."""
        if mido is None:
            raise RuntimeError(f"mido unavailable: {_MIDO_ERROR}")
        with mido.open_input(midi_in) as inp, mido.open_output(midi_out) as out:
            for msg in inp:
                events = await asyncio.to_thread(self._trigger, msg)
                for ev in events:
                    pitch = int(ev.get("pitch", 36))
                    vel = int(ev.get("velocity", 100))
                    out.send(mido.Message("note_on", note=pitch, velocity=vel))
                    dur = float(ev.get("duration", 0.5))
                    await asyncio.sleep(dur * 60.0 / self.bpm)
                    out.send(mido.Message("note_off", note=pitch))

class TransformerInteractiveEngine:
    """Generate bass lines using :class:`BassTransformer`."""

    def __init__(
        self,
        model_name: str = "gpt2-medium",
        *,
        bpm: float = 120.0,
        buffer_ms: int = 100,
        lookahead_bars: float = 0.5,
    ) -> None:
        self.model = BassTransformer(model_name)
        self.bpm = bpm
        self.buffer_ms = buffer_ms
        self.lookahead_bars = lookahead_bars
        self.callbacks: list[Callable[[dict], None]] = []
        self._seq: list[int] = [0]
        self._buf = LiveBuffer(self._gen_bar, buffer_ahead=2)

    def add_callback(self, callback: Callable[[dict], None]) -> None:
        self.callbacks.append(callback)

    def _gen_bar(self, _idx: int) -> list[dict]:
        tokens = self.model.sample(self._seq[-16:], top_k=8, temperature=1.0)
        self._seq.extend(tokens)
        return [
            {
                "pitch": int(tok),
                "velocity": 100,
                "offset": i * 0.25,
                "duration": 0.25,
            }
            for i, tok in enumerate(tokens)
        ]

    def _trigger(self, msg: Any) -> list[dict]:
        if msg.type == "note_on" and msg.velocity > 0:
            self._seq.extend(self.model.encode([{"pitch": msg.note}]))
            events: list[dict] = []
            for _ in range(2):
                events.extend(self._buf.get_next())
            for cb in self.callbacks:
                for ev in events:
                    cb(ev)
            return events
        return []

    async def start(self, midi_in: str, midi_out: str) -> None:
        if mido is None:
            raise RuntimeError(f"mido unavailable: {_MIDO_ERROR}")
        with mido.open_input(midi_in) as inp, mido.open_output(midi_out) as out:
            for msg in inp:
                events = await asyncio.to_thread(self._trigger, msg)
                for ev in events:
                    pitch = int(ev.get("pitch", 36))
                    vel = int(ev.get("velocity", 100))
                    out.send(mido.Message("note_on", note=pitch, velocity=vel))
                    dur = float(ev.get("duration", 0.5))
                    await asyncio.sleep(dur * 60.0 / self.bpm)
                    out.send(mido.Message("note_off", note=pitch))

__all__ = ["InteractiveEngine", "TransformerInteractiveEngine"]
