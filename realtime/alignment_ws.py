from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, WebSocket

from scripts.align_lyrics import align_audio, load_model

app = FastAPI()
_model: tuple[Any, dict, list[str]] | None = None
_lock = asyncio.Lock()


@app.post("/warmup")
async def warmup(ckpt: str) -> dict[str, str]:
    global _model
    if _lock.locked():
        return {"status": "busy"}
    async with _lock:
        _model = await asyncio.to_thread(load_model, Path(ckpt))
    return {"status": "ok"}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.websocket("/infer")
async def infer(ws: WebSocket, chunk_ms: int | None = None) -> None:
    await ws.accept()
    init = await ws.receive_text()
    midi_times = json.loads(init).get("midi", [])
    buf = bytearray()
    limit = 1_000_000
    model, cfg, vocab = _model if _model else (None, {}, [])
    chunk = (
        int(chunk_ms * cfg.get("sample_rate", 16000) / 1000) * 4 if chunk_ms else None
    )
    while True:
        data = await ws.receive_bytes()
        if not data:
            break
        if len(data) > limit:
            await ws.send_json({"error": "payload too large"})
            await ws.close(code=1011)
            break
        buf.extend(data)
        model, cfg, vocab = _model or (None, {}, [])
        if model is None:
            await ws.send_json([])
            buf.clear()
            continue
        while chunk is None or len(buf) >= chunk:
            if not buf:
                break
            segment = np.frombuffer(buf[: chunk or len(buf)], dtype=np.float32)
            del buf[: chunk or len(buf)]
            try:
                res = await asyncio.to_thread(
                    align_audio,
                    torch.tensor(segment),
                    midi_times,
                    model,
                    cfg,
                    vocab,
                )
            except Exception as exc:  # pragma: no cover - runtime safety
                await ws.send_json({"error": str(exc)})
                await ws.close(code=1011)
                return
            else:
                await ws.send_json(res)
                if cfg.get("heartbeat"):
                    await ws.send_json({"heartbeat": True})
            if chunk is None:
                break


__all__ = ["app", "warmup", "infer"]
