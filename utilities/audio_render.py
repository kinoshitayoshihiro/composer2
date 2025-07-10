from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Mapping

from music21 import stream

from .convolver import render_wav
from .mix_profile import get_mix_chain


def render_part_audio(
    part: stream.Part | Mapping[str, stream.Part],
    *,
    ir_name: str | None = None,
    out_path: str | Path | None = "out.wav",
    sf2: str | None = None,
    quality: str = "fast",
    bit_depth: int = 24,
    oversample: int = 1,
    normalize: bool = True,
    dither: bool = True,
    tail_db_drop: float = -60.0,
    **mix_opts,
) -> Path:
    """Render ``part`` to ``out_path`` applying ``ir_name`` if given."""

    if isinstance(part, dict):
        parts = part
    else:
        parts = {getattr(part, "id", "part"): part}

    score = stream.Score()
    for p in parts.values():
        score.insert(0, p)
    tmp_mid = NamedTemporaryFile(suffix=".mid", delete=False)
    score.write("midi", fp=tmp_mid.name)

    if ir_name is None:
        meta = next(iter(parts.values())).metadata
        ir_file = getattr(meta, "ir_file", None) if meta is not None else None
    else:
        p = Path(ir_name)
        if p.is_file():
            ir_file = str(p)
        else:
            chain = get_mix_chain(ir_name, {}) or {}
            ir_file = chain.get("ir_file")

    if out_path is None:
        out_path = "out.wav"

    if bit_depth not in (16, 24, 32):
        raise ValueError("bit_depth must be 16, 24, or 32")

    out = render_wav(
        tmp_mid.name,
        ir_file or "",
        str(out_path),
        sf2=sf2,
        parts=parts,
        quality=quality,
        bit_depth=bit_depth,
        oversample=oversample,
        normalize=normalize,
        dither=dither,
        tail_db_drop=tail_db_drop,
        **mix_opts,
    )

    Path(tmp_mid.name).unlink(missing_ok=True)
    for p in parts.values():
        if p.metadata is not None:
            setattr(p.metadata, "rendered_wav", str(out))
    return out
