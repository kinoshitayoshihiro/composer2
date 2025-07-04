import shutil
import soundfile as sf
from pathlib import Path
from . import mix_profile


def render_ir(midi_path: str, output_wav: str, *, ir_path: str, mix_preset: str | None = None) -> None:
    """Render *midi_path* using the given IR file and write to *output_wav*.

    This is a simplified offline renderer that copies the IR as placeholder
    audio. A real implementation would convolve the synthesized audio with the
    impulse response via ffmpeg or sox.
    """

    ir_file = Path(ir_path)
    if not ir_file.is_file():
        raise FileNotFoundError(ir_file)
    chain = mix_profile.get_mix_chain(mix_preset or "default", {})
    _ = chain  # placeholder for future processing
    shutil.copy(ir_file, output_wav)

__all__ = ["render_ir"]
