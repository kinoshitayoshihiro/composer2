from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Mapping, Literal, Any
import warnings

from music21 import stream

from .convolver import render_wav
from .mix_profile import get_mix_chain


def render_part_audio(
    part: stream.Part | Mapping[str, stream.Part],
    *,
    downmix: Literal["auto", "stereo", "none"] = "auto",
    normalize: bool = True,
    dither: bool = True,
    **kw: Any,
) -> Path:
    """Render ``part`` to ``out_path`` applying ``ir_name`` if given."""

    if "mix_opts" in kw:
        warnings.warn(
            "'mix_opts' dict is deprecated; pass options as keyword arguments",
            DeprecationWarning,
            stacklevel=2,
        )
        mix_opts = kw.pop("mix_opts") or {}
        if isinstance(mix_opts, Mapping):
            kw.update(mix_opts)

    ir_name = kw.pop("ir_name", None)
    out_path = kw.pop("out_path", "out.wav")
    sf2 = kw.pop("sf2", None)
    quality = kw.pop("quality", "fast")
    bit_depth = kw.pop("bit_depth", 24)
    oversample = kw.pop("oversample", 1)
    tail_db_drop = kw.pop("tail_db_drop", -60.0)

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
        downmix=downmix,
        tail_db_drop=tail_db_drop,
        **kw,
    )

    Path(tmp_mid.name).unlink(missing_ok=True)
    for p in parts.values():
        if p.metadata is not None:
            setattr(p.metadata, "rendered_wav", str(out))
    return out
