from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


def render_midi(midi_path: str | Path, out_wav: str | Path, sf2_path: str | Path | None = None) -> Path:
    """Render ``midi_path`` to ``out_wav`` using ``fluidsynth``.

    If ``sf2_path`` is ``None`` the environment variable ``SF2_PATH`` is used.
    Raises ``RuntimeError`` if ``fluidsynth`` or the SoundFont is missing.
    """
    fs_bin = shutil.which("fluidsynth")
    if not fs_bin:
        raise RuntimeError("fluidsynth executable not found")

    soundfont = sf2_path or os.environ.get("SF2_PATH")
    if not soundfont or not Path(soundfont).exists():
        raise RuntimeError("SoundFont path not provided or does not exist")

    midi_path = Path(midi_path)
    out_wav = Path(out_wav)

    cmd = [
        fs_bin,
        "-ni",
        str(soundfont),
        str(midi_path),
        "-F",
        str(out_wav),
    ]
    subprocess.run(cmd, check=True)
    return out_wav
