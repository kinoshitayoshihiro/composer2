from __future__ import annotations

from pathlib import Path

from pydantic import FilePath
from pydantic_settings import BaseSettings


class ArticSettings(BaseSettings):
    schema_path: FilePath = (
        Path(__file__).resolve().parent.parent / "articulation_schema.yaml"
    )


settings = ArticSettings()
