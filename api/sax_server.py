from __future__ import annotations

"""Simple FastAPI server exposing sax solo generation."""

from types import ModuleType
from typing import Any, List

from utilities.fastapi_compat import FastAPI, HTTPException, Request, JSONResponse
from pydantic import BaseModel

_PLUGIN_MODULE: ModuleType | None = None
_PLUGIN_ERROR: ImportError | None = None


def _load_plugin() -> ModuleType:
    global _PLUGIN_MODULE, _PLUGIN_ERROR
    if _PLUGIN_MODULE is not None:
        return _PLUGIN_MODULE
    if _PLUGIN_ERROR is not None:
        raise _PLUGIN_ERROR
    try:
        import plugins.sax_companion_plugin as sax_plugin  # type: ignore
    except ImportError as exc:
        try:
            import plugins.sax_companion_stub as sax_plugin  # type: ignore
        except ImportError as stub_exc:  # pragma: no cover - both missing
            _PLUGIN_ERROR = exc
            raise stub_exc from exc
    _PLUGIN_MODULE = sax_plugin
    return sax_plugin

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
def generate_sax(req: SaxRequest) -> List[dict[str, Any]]:
    """Return sax notes using plugin or stub."""
    try:
        plugin = _load_plugin()
    except ImportError as exc:  # pragma: no cover - propagated in tests
        raise HTTPException(status_code=404, detail="sax plugin unavailable") from exc

    payload = req.model_dump()
    try:
        notes = plugin.generate_notes(payload)
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - runtime plugin errors
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if isinstance(notes, list) and notes and isinstance(notes[0], dict):
        first = notes[0]
        if "error" in first:
            message = str(first.get("message") or first["error"])
            raise HTTPException(status_code=400, detail=message)

    if not isinstance(notes, list):
        raise HTTPException(status_code=500, detail="invalid plugin response")
    return notes
