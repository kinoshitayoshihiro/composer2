from __future__ import annotations

"""Simple FastAPI server exposing sax solo generation."""

from fastapi import FastAPI
from pydantic import BaseModel

try:
    import plugins.sax_companion_plugin as sax_plugin
except Exception:  # pragma: no cover - plugin not built
    import plugins.sax_companion_stub as sax_plugin  # type: ignore

app = FastAPI()


class SaxRequest(BaseModel):
    growl: bool = False
    altissimo: bool = False


@app.post("/generate_sax")
def generate_sax(req: SaxRequest) -> list[dict[str, object]]:
    """Return sax notes using plugin or stub."""
    return sax_plugin.generate_notes(req.model_dump())
