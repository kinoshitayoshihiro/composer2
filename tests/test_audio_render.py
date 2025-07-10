import importlib.util
import types
import numpy as np
from pathlib import Path

if importlib.util.find_spec("soundfile") is None:
    sf = types.SimpleNamespace(
        write=lambda p, d, sr, subtype=None: Path(p).write_bytes(b""),
        info=lambda p: types.SimpleNamespace(samplerate=44100, subtype="PCM_16"),
        read=lambda p: (np.zeros(1, dtype=np.float32), 44100),
    )
else:
    import soundfile as sf

from music21 import stream, note

from utilities.audio_render import render_part_audio


def _make_part():
    p = stream.Part()
    p.append(note.Note('C4', quarterLength=1))
    return p


import pytest


@pytest.mark.parametrize("soxr_present", [True, False])
def test_render_part_audio_options(tmp_path, monkeypatch, soxr_present):
    import utilities.convolver as conv
    if not soxr_present:
        monkeypatch.setattr(conv, "soxr", None)
    part = _make_part()
    for q in ["high", "ultra"]:
        for b in [16, 24, 32]:
            out = tmp_path / f"out_{q}_{b}.wav"
            render_part_audio(part, out_path=out, quality=q, bit_depth=b, oversample=4)
            info = sf.info(out)
            assert info.samplerate == 44100
            subtype = "FLOAT" if b == 32 else f"PCM_{b}"
            assert info.subtype == subtype
            assert out.is_file()
