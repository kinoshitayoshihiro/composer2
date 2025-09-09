import shutil


def has_fluidsynth() -> bool:
    """Return True if fluidsynth is available on PATH."""
    return shutil.which("fluidsynth") is not None
