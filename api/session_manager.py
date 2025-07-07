from __future__ import annotations

import base64
import io
from typing import Dict, List, Set

import pretty_midi
from fastapi import WebSocket


class Session:
    def __init__(self, model: str, tempo: int) -> None:
        self.model = model
        self.tempo = tempo
        self.users: Set[WebSocket] = set()


class SessionManager:
    """Manage websocket sessions and MIDI generation."""

    def __init__(self) -> None:
        self.sessions: Dict[str, Session] = {}

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

    async def join(self, session_id: str, ws: WebSocket) -> None:
        await ws.accept()
        sess = self.sessions.setdefault(session_id, Session(model=session_id, tempo=120))
        sess.users.add(ws)

    def leave(self, session_id: str, ws: WebSocket) -> None:
        sess = self.sessions.get(session_id)
        if not sess:
            return
        sess.users.discard(ws)
        if not sess.users:
            self.sessions.pop(session_id, None)

    async def broadcast(self, session_id: str, midi_bytes: bytes) -> None:
        sess = self.sessions.get(session_id)
        if not sess:
            return
        payload = {"midi": base64.b64encode(midi_bytes).decode()}
        for ws in list(sess.users):
            await ws.send_json(payload)

