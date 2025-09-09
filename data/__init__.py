"""Data module for articulation tagger datasets."""

from importlib import resources as _resources
from typing import IO

__all__ = ["open_text"]


def open_text(name: str) -> IO[str]:
    return _resources.files(__name__).joinpath(name).open("r")
