from __future__ import annotations

import random
from pathlib import Path

import pretty_midi


def swing(pm: pretty_midi.PrettyMIDI, amount: float) -> pretty_midi.PrettyMIDI:
    """Apply swing by delaying off-beat eighths by ``amount`` percent."""
    tempo = pm.get_tempo_changes()[1]
    bpm = float(tempo[0]) if getattr(tempo, "size", 0) else 120.0
    beat = 60.0 / bpm
    offset = beat * 0.5 * (amount / 100)
    for inst in pm.instruments:
        for note in inst.notes:
            beat_pos = note.start / beat
            if abs((beat_pos % 1) - 0.5) < 1e-3:
                note.start += offset
                note.end += offset
    return pm


def shuffle(pm: pretty_midi.PrettyMIDI, seed: int | None = None) -> pretty_midi.PrettyMIDI:
    rng = random.Random(seed)
    for inst in pm.instruments:
        rng.shuffle(inst.notes)
    return pm


def transpose(pm: pretty_midi.PrettyMIDI, semitone: int) -> pretty_midi.PrettyMIDI:
    for inst in pm.instruments:
        for note in inst.notes:
            note.pitch += semitone
    return pm


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - CLI entry
    import argparse

    ap = argparse.ArgumentParser(prog="modcompose augment")
    ap.add_argument("midi", type=Path)
    ap.add_argument("--swing", type=float, default=0.0)
    ap.add_argument("--transpose", type=int, default=0)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("-o", "--out", type=Path, required=True)
    ns = ap.parse_args(argv)
    pm = pretty_midi.PrettyMIDI(str(ns.midi))
    if ns.swing:
        pm = swing(pm, ns.swing)
    if ns.shuffle:
        pm = shuffle(pm)
    if ns.transpose:
        pm = transpose(pm, ns.transpose)
    pm.write(str(ns.out))
    print(f"wrote {ns.out}")


if __name__ == "__main__":  # pragma: no cover - manual
    main()
