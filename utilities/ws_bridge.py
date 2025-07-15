from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

from .ring_buffer import RingBuffer

try:
    import websockets
except Exception:  # pragma: no cover - optional
    websockets = None  # type: ignore

try:
    from .rt_midi_streamer import RtMidiStreamer
except Exception:  # pragma: no cover - optional
    RtMidiStreamer = None  # type: ignore


def _parse_args(args: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Realtime WebSocket bridge")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--buffer", type=int, default=256)
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--midi-port", type=str, default=None)
    return ap.parse_args(args)


class _Bridge:
    def __init__(self, buf_size: int, midi_port: str | None = None) -> None:
        self.buffer = RingBuffer(buf_size)
        self.midi: RtMidiStreamer | None = None
        if midi_port and RtMidiStreamer is not None:
            try:
                self.midi = RtMidiStreamer(midi_port, buffer_ms=0)
            except Exception:
                self.midi = None

    async def handle(self, ws: Any) -> None:
        async for msg in ws:
            try:
                data = json.loads(msg)
            except Exception:
                continue
            if data.get("type") == "control":
                continue
            if self.midi is not None and data.get("type") == "note":
                midi = int(data.get("midi", 60))
                vel = int(data.get("vel", 100))
                await asyncio.to_thread(self.midi._send_with_retry, [0x90, midi, vel])
            await ws.send(json.dumps(data))


async def serve(port: int = 8765, buf_size: int = 256, midi_port: str | None = None) -> Any:
    if websockets is None:
        raise RuntimeError("websockets library required")
    bridge = _Bridge(buf_size, midi_port)
    return await websockets.serve(bridge.handle, "localhost", port, process_request=None)


def main(argv: list[str] | None = None) -> None:
    opts = _parse_args(argv)
    server = asyncio.run(serve(opts.port, opts.buffer, opts.midi_port))
    print("Streaming started")
    try:
        asyncio.get_event_loop().run_until_complete(server.wait_closed())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
