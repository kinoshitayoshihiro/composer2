from __future__ import annotations

import base64
import io
import asyncio
from typing import Dict, List, Set, Optional

import pretty_midi
from utilities.fastapi_compat import WebSocket
try:
    import redis.asyncio as redis  # type: ignore
except Exception:  # pragma: no cover - redis optional
    redis = None


class Session:
    def __init__(self, model: str, tempo: int) -> None:
        self.model = model
        self.tempo = tempo
        self.users: Set[WebSocket] = set()


class SessionManager:
    """Manage websocket sessions and MIDI generation using Redis pub/sub."""

    def __init__(self, redis_url: str = "redis://localhost:6379") -> None:
        self.sessions: Dict[str, Session] = {}
        self._tasks: Dict[str, asyncio.Task] = {}
        self.redis_url = redis_url
        self.redis: Optional[redis.Redis] = None
        if redis is not None:
            try:
                self.redis = redis.from_url(redis_url)
            except Exception:
                self.redis = None

    def generate(self, model_id: str, chords: List[int], bars: int, tempo: int = 120) -> bytes:
        pm = pretty_midi.PrettyMIDI()
        inst = pretty_midi.Instrument(program=0)
        beat = 60.0 / tempo
        for b in range(bars):
            start_bar = b * 4 * beat
            for i, pitch in enumerate(chords):
                start = start_bar + i * beat
                note = pretty_midi.Note(velocity=100, pitch=int(pitch), start=start, end=start + beat)
                inst.notes.append(note)
        pm.instruments.append(inst)
        buf = io.BytesIO()
        pm.write(buf)
        return buf.getvalue()

    async def _subscriber(self, session_id: str, sess: Session) -> None:
        if not self.redis:
            return
        pubsub = self.redis.pubsub()
        await pubsub.subscribe(session_id)
        try:
            async for msg in pubsub.listen():
                if msg.get("type") != "message":
                    continue
                try:
                    data = base64.b64decode(msg["data"])  # type: ignore[index]
                except Exception:
                    continue
                await self._broadcast_local(session_id, data)
        finally:
            await pubsub.unsubscribe(session_id)

    async def join(self, session_id: str, ws: WebSocket) -> None:
        await ws.accept()
        sess = self.sessions.setdefault(session_id, Session(model=session_id, tempo=120))
        sess.users.add(ws)
        if session_id not in self._tasks and self.redis is not None:
            self._tasks[session_id] = asyncio.create_task(self._subscriber(session_id, sess))

    def leave(self, session_id: str, ws: WebSocket) -> None:
        sess = self.sessions.get(session_id)
        if not sess:
            return
        sess.users.discard(ws)
        if not sess.users:
            self.sessions.pop(session_id, None)
            task = self._tasks.pop(session_id, None)
            if task:
                task.cancel()

    async def broadcast(self, session_id: str, midi_bytes: bytes) -> None:
        if self.redis is not None:
            await self.redis.publish(session_id, base64.b64encode(midi_bytes).decode())
        await self._broadcast_local(session_id, midi_bytes)

    async def _broadcast_local(self, session_id: str, midi_bytes: bytes) -> None:
        sess = self.sessions.get(session_id)
        if not sess:
            return
        payload = {"midi": base64.b64encode(midi_bytes).decode()}
        for ws in list(sess.users):
            await ws.send_json(payload)

