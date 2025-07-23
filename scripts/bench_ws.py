from __future__ import annotations

import asyncio
import time

import uvicorn
import websockets

from realtime.phrase_ws import app, run_server

DUMMY_MIDI = bytes.fromhex(
    "4d54686400000006000100010060" "4d54726b0000000400ff2f00" * 4
)


async def bench(n: int = 20, host: str = "127.0.0.1", port: int = 8765) -> float:
    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    task = asyncio.create_task(asyncio.to_thread(run_server, server))
    await asyncio.sleep(0.1)
    uri = f"ws://{host}:{port}/infer"
    start = time.perf_counter()
    async with websockets.connect(uri, ping_interval=None) as ws:
        for _ in range(n):
            await ws.send(DUMMY_MIDI)
            await ws.recv()
    avg = (time.perf_counter() - start) * 1000 / n
    server.should_exit = True
    await task
    print(f"avg {avg:.1f} ms")
    return avg


if __name__ == "__main__":  # pragma: no cover - CLI
    asyncio.run(bench())
