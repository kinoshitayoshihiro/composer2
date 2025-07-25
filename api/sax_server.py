from __future__ import annotations

"""Simple FastAPI server exposing sax solo generation."""

try:  # pragma: no cover - prefer actual FastAPI when available
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse
except Exception:  # pragma: no cover - optional dependency missing
    from starlette.applications import Starlette as FastAPI
    from starlette.exceptions import HTTPException
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request
    from starlette.responses import JSONResponse

    class FastAPIStub(FastAPI):  # type: ignore
        def post(self, path):
            def decorator(func):
                self.add_route(path, func, methods=["POST"])
                return func

            return decorator

        def middleware(self, _name):
            def decorator(func):
                self.add_middleware(BaseHTTPMiddleware, dispatch=func)
                return func

            return decorator

    FastAPI = FastAPIStub  # type: ignore
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
