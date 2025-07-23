from __future__ import annotations

import asyncio
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, WebSocket

from scripts.segment_phrase import load_model, segment_bytes

app = FastAPI()
_model: torch.nn.Module | None = None
_lock = asyncio.Lock()


@app.post("/warmup")  # type: ignore[misc]
async def warmup(
    arch: str = "transformer", ckpt: str = "phrase.ckpt"
) -> dict[str, str]:
    global _model
    if _lock.locked():
        return {"status": "busy"}
    async with _lock:
        _model = await asyncio.to_thread(load_model, arch, Path(ckpt))
    return {"status": "ok"}


@app.websocket("/infer")  # type: ignore[misc]
async def infer(ws: WebSocket) -> None:
    await ws.accept()
    while True:
        data = await ws.receive_bytes()
        if not data:
            break
        if len(data) > 5_000_000:
            await ws.send_json({"error": "payload_too_large"})
            continue
        if _model is None:
            await ws.send_json([])
            continue
        res = segment_bytes(data, _model, 0.5)
        await ws.send_json(res)


def run_server(server: uvicorn.Server) -> None:
    """Run *server* within a new event loop."""
    asyncio.run(server.serve())


def run(host: str = "127.0.0.1", port: int = 8000) -> None:
    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    run_server(server)


__all__ = ["run", "run_server", "app"]
