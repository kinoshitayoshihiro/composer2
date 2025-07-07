from __future__ import annotations

import asyncio
import json
import threading
from typing import Any, List

try:
    import websockets
except Exception:  # pragma: no cover - optional
    websockets = None  # type: ignore

try:
    from generator.piano_ml_generator import PianoMLGenerator
except Exception:  # pragma: no cover - optional
    PianoMLGenerator = None  # type: ignore


class WarmModel:
    """Load ``PianoMLGenerator`` in a background thread and serve predictions."""

    def __init__(self, model_path: str = "piano_model") -> None:
        self._lock = threading.Lock()
        self._generator: Any | None = None
        self.model_path = model_path
        thread = threading.Thread(target=self._load, daemon=True)
        thread.start()

    def _load(self) -> None:
        if PianoMLGenerator is None:
            return
        gen = PianoMLGenerator(self.model_path)
        # warm the model by generating once
        gen.generate(max_bars=1)
        with self._lock:
            self._generator = gen

    def step(self, chord: List[int], bars_context: int) -> List[int]:
        with self._lock:
            gen = self._generator
        if gen is None:
            return chord[:16]
        events = [{"pitch": int(p)} for p in chord]
        for _ in range(bars_context):
            gen.step(events)
        out = gen.step(events)
        return [int(ev.get("pitch", 0)) for ev in out]


async def serve(model: WarmModel, host: str = "localhost", port: int = 8765):
    if websockets is None:
        raise RuntimeError("websockets library required")

    async def handler(ws):
        async for message in ws:
            data = json.loads(message)
            chord = data.get("chord", [])
            bars = int(data.get("bars_context", 2))
            tokens = await asyncio.to_thread(model.step, chord, bars)
            await ws.send(json.dumps(tokens))

    return await websockets.serve(handler, host, port)


async def main() -> None:
    model = WarmModel()
    server = await serve(model)
    await server.wait_closed()


__all__ = ["WarmModel", "serve", "main"]
