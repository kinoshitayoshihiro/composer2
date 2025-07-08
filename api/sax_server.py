from __future__ import annotations

"""Simple FastAPI server exposing sax solo generation."""

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

try:
    import plugins.sax_companion_plugin as sax_plugin
except Exception:  # pragma: no cover - plugin not built
    import plugins.sax_companion_stub as sax_plugin  # type: ignore

app = FastAPI()


@app.middleware("http")
async def handle_errors(request: Request, call_next):
    try:
        return await call_next(request)
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - runtime logging
        return JSONResponse(status_code=500, content={"detail": str(exc)})


class SaxRequest(BaseModel):
    growl: bool = False
    altissimo: bool = False

    class Config:
        extra = "forbid"


@app.post("/generate_sax")
def generate_sax(req: SaxRequest) -> list[dict[str, object]]:
    """Return sax notes using plugin or stub."""
    notes = sax_plugin.generate_notes(req.model_dump())
    if (
        isinstance(notes, list)
        and notes
        and isinstance(notes[0], dict)
        and "error" in notes[0]
    ):
        raise HTTPException(status_code=500, detail=str(notes[0].get("message", "")))
    return notes
