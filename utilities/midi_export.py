from __future__ import annotations

from pathlib import Path

import pretty_midi


def write_demo_bar(path: str | Path) -> Path:
    """Write a deterministic 4-beat MIDI bar for tests."""
    out_path = Path(path)
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=0)
    qlen = 0.5  # quarter note length in seconds at 120 BPM
    for i in range(4):
        start = i * qlen
        note = pretty_midi.Note(velocity=90, pitch=60, start=start, end=start + qlen)
        inst.notes.append(note)
    pm.instruments.append(inst)
    pm.write(str(out_path))
    return out_path
