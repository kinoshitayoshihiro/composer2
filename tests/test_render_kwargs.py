import hashlib
import warnings
from pathlib import Path
from music21 import stream, note
from utilities import audio_render


def _make_part():
    p = stream.Part()
    p.append(note.Note('C4', quarterLength=1))
    return p


def test_render_kwargs_deprecated_dict(tmp_path, monkeypatch):
    part = _make_part()

    def fake_render_wav(_mid, _ir, out, **_kw):
        Path(out).write_bytes(b'X')
        return Path(out)

    monkeypatch.setattr(audio_render, "render_wav", fake_render_wav)

    out1 = tmp_path / "a.wav"
    out2 = tmp_path / "b.wav"

    audio_render.render_part_audio(part, out_path=out1, quality="fast", bit_depth=24)
    with warnings.catch_warnings(record=True) as w:
        audio_render.render_part_audio(
            part,
            mix_opts={"out_path": out2, "quality": "fast", "bit_depth": 24},
        )
        assert any(issubclass(x.category, DeprecationWarning) for x in w)

    h1 = hashlib.sha1(out1.read_bytes()).hexdigest()
    h2 = hashlib.sha1(out2.read_bytes()).hexdigest()
    assert h1 == h2
