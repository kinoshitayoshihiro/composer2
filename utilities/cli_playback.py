"""Utilities for CLI MIDI playback."""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Callable type used for playback functions
PlayFunc = Callable[[bytes], None]


def _linux_player() -> Optional[PlayFunc]:
    """Return player function for Linux."""
    for cmd in ("timidity", "fluidsynth", "aplaymidi"):
        path = shutil.which(cmd)
        if not path:
            continue
        if cmd == "fluidsynth":
            sf2 = os.environ.get("SF2_PATH")
            if not sf2 or not Path(sf2).exists():
                continue
            def play(data: bytes, path=path, sf2=sf2) -> None:
                subprocess.run([path, "-ni", sf2, "-"], input=data, check=False)
            return play
        if cmd == "aplaymidi":
            def play(data: bytes, path=path) -> None:
                with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
                    tmp.write(data)
                    tmp.flush()
                    fname = tmp.name
                try:
                    subprocess.run([path, fname], check=False)
                finally:
                    try:
                        os.unlink(fname)
                    except OSError:
                        pass
            return play
        # timidity
        def play(data: bytes, path=path) -> None:
            subprocess.run([path, "-"], input=data, check=False)
        return play
    return None


def _macos_player() -> Optional[PlayFunc]:
    """Return player function for macOS."""
    afplay = shutil.which("afplay")
    if not afplay:
        return None
    try:
        from midi2audio import FluidSynth  # type: ignore
        fs = FluidSynth()
    except Exception:  # pragma: no cover - optional dependency
        fs = None

    def play(data: bytes, player=afplay, fs=fs) -> None:
        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp_mid:
            tmp_mid.write(data)
            tmp_mid.flush()
            mid_path = tmp_mid.name
        wav_path = None
        try:
            if fs is not None:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
                    wav_path = tmp_wav.name
                fs.midi_to_audio(mid_path, wav_path)
                subprocess.run([player, wav_path], check=False)
            else:
                logger.warning("midi2audio not installed; playing raw MIDI")
                subprocess.run([player, mid_path], check=False)
        finally:
            try:
                os.unlink(mid_path)
            except OSError:
                pass
            if wav_path:
                try:
                    os.unlink(wav_path)
                except OSError:
                    pass
    return play


def _windows_player() -> Optional[PlayFunc]:
    """Return player function for Windows."""
    for exe in ("wmplayer", "start"):
        path = shutil.which(exe)
        if path:
            def play(data: bytes, path=path, shell=exe == "start") -> None:
                with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
                    tmp.write(data)
                    tmp.flush()
                    fname = tmp.name
                try:
                    subprocess.run([path, fname], shell=shell, check=False)
                finally:
                    try:
                        os.unlink(fname)
                    except OSError:
                        pass
            return play
    pwsh = shutil.which("powershell")
    if pwsh:
        def play(data: bytes, pwsh=pwsh) -> None:
            with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
                tmp.write(data)
                tmp.flush()
                fname = tmp.name
            try:
                cmd = [pwsh, "-c", f'(New-Object Media.SoundPlayer \"{fname}\").PlaySync()']
                subprocess.run(cmd, check=False)
            finally:
                try:
                    os.unlink(fname)
                except OSError:
                    pass
        return play
    return None


def find_player() -> Optional[PlayFunc]:
    """Return a function that plays MIDI bytes or ``None`` if none available."""
    if sys.platform.startswith("linux"):
        return _linux_player()
    if sys.platform == "darwin":
        return _macos_player()
    if sys.platform.startswith("win"):
        return _windows_player()
    return None

