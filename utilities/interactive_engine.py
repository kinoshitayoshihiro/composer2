from __future__ import annotations

from collections.abc import Callable
from typing import Any
import warnings

try:
    import mido
except Exception as e:  # pragma: no cover - optional
    mido = None  # type: ignore
    _MIDO_ERROR = e
else:
    _MIDO_ERROR = None

from .ai_sampler import TransformerBassGenerator


class InteractiveEngine:
    """Trigger bass generation on incoming MIDI events."""

    def __init__(self, model_name: str = "gpt2-music", bpm: float = 120.0) -> None:
        self.generator = TransformerBassGenerator(model_name)
        self.bpm = bpm
        self.callbacks: list[Callable[[dict], None]] = []

    def add_callback(self, callback: Callable[[dict], None]) -> None:
        self.callbacks.append(callback)

    def _trigger(self, msg: Any) -> None:
        if msg.type == "note_on" and msg.velocity > 0:
            events = self.generator.generate([
                {"pitch": msg.note, "velocity": msg.velocity}
            ], 1)
            for cb in self.callbacks:
                for ev in events:
                    cb(ev)

    def run(self) -> None:
        if mido is None:
            raise RuntimeError(f"mido unavailable: {_MIDO_ERROR}")
        names = mido.get_input_names()
        if not names:
            warnings.warn("No MIDI input ports available", RuntimeWarning)
            return
        with mido.open_input(names[0]) as port:
            for msg in port:
                self._trigger(msg)

__all__ = ["InteractiveEngine"]
