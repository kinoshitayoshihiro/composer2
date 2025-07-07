import numpy as np
from music21 import note, volume

from utilities.loudness_normalizer import normalize_velocities


def _lufs(vals):
    rms = np.sqrt(np.mean(np.array(vals, dtype=float) ** 2)) or 1e-9
    return 20 * np.log10(rms / 127.0)


def test_velocity_normalizer_reduces_rms() -> None:
    notes = [note.Note("C4") for _ in range(16)]
    for n in notes:
        n.volume = volume.Volume(velocity=120)
    before = _lufs([n.volume.velocity for n in notes])
    after = normalize_velocities(notes, target_lufs=-16)
    diff = before - after
    assert diff > 3
