from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from music21 import stream


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
        default_sf = Path("/usr/share/sounds/sf2/FluidR3_GM.sf2")
        if default_sf.exists():
            soundfont = str(default_sf)
        else:
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


def export_audio(
    midi_path: str | Path,
    out_wav: str | Path,
    *,
    soundfont: str | Path | None = None,
    ir_file: str | Path | None = None,
    part: "stream.Part" | None = None,
) -> Path:
    """Render ``midi_path`` and optionally convolve with ``ir_file``."""
    wav = render_midi(midi_path, out_wav, sf2_path=soundfont)
    if ir_file is None and part is not None and hasattr(part, "metadata"):
        ir_file = getattr(part.metadata, "ir_file", None)
    if ir_file:
        from .convolver import render_with_ir

        render_with_ir(wav, ir_file, out_wav)
    return Path(out_wav)
