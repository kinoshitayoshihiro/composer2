import io
import random
from pathlib import Path

import pytest

try:
    import pretty_midi
    import torch  # noqa: F401
except Exception as exc:  # pragma: no cover - optional
    pytest.skip(f"deps missing: {exc}", allow_module_level=True)

from scripts.segment_phrase import load_model, segment_bytes

CKPT = Path("checkpoints/epoch=0-step=2.ckpt")


def _make_midi(rng: random.Random) -> bytes:
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0)
    for i in range(16):
        start = i * 0.5 + rng.uniform(-5 / 480.0, 5 / 480.0)
        inst.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=start, end=start + 0.25))
    pm.instruments.append(inst)
    buf = io.BytesIO()
    pm.write(buf)
    return buf.getvalue()


def test_segment_noise_stability() -> None:
    model = load_model("transformer", CKPT)
    rng = random.Random(0)
    counts = []
    for _ in range(30):
        data = _make_midi(rng)
        seg = segment_bytes(data, model, 0.5)
        counts.append(len(seg))
    assert max(counts) - min(counts) <= 1

