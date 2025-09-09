from __future__ import annotations

import base64
from typing import List

from utilities.fastapi_compat import (
    FastAPI,
    WebSocket,
    WebSocketDisconnect,
    HTTPException,
    Request,
    status,
    JSONResponse,
)
from pydantic import BaseModel

from .session_manager import SessionManager

app = FastAPI()
manager = SessionManager()


@app.middleware("http")
async def handle_errors(request: Request, call_next):
    try:
        return await call_next(request)
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - runtime logging
        return JSONResponse(status_code=500, content={"detail": str(exc)})


class GenerateRequest(BaseModel):
    model_id: str
    chords: List[int]
    bars: int


class GenerateResponse(BaseModel):
    midi: str


@app.post("/generate_vocal", response_model=GenerateResponse)
async def generate_vocal(req: GenerateRequest) -> GenerateResponse:
    data = manager.generate(req.model_id, req.chords, req.bars)
    return GenerateResponse(midi=base64.b64encode(data).decode())


@app.websocket("/session/vocal/{session_id}")
async def session_ws(websocket: WebSocket, session_id: str) -> None:
    await manager.join(session_id, websocket)
    try:
        while True:
            payload = await websocket.receive_json()
            chords = payload.get("chords", [])
            bars = int(payload.get("bars", 1))
            data = manager.generate(session_id, chords, bars)
            await manager.broadcast(session_id, data)
    except WebSocketDisconnect:
        manager.leave(session_id, websocket)
