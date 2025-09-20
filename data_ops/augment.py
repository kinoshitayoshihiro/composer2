from __future__ import annotations

import random
from pathlib import Path

import pretty_midi


def swing(pm: pretty_midi.PrettyMIDI, ratio: float) -> pretty_midi.PrettyMIDI:
    """Apply swing feel by moving off-beats by ``(ratio-50) %`` of a 16th."""
    tempo = pm.get_tempo_changes()[1]
    bpm = float(tempo[0]) if getattr(tempo, "size", 0) else 120.0
    if bpm <= 0:
        bpm = 120.0
    beat = 60.0 / bpm
    offset = (ratio - 50.0) / 100.0 * (beat / 6)
    for inst in pm.instruments:
        for note in inst.notes:
            beat_pos = note.start / beat
            if abs((beat_pos % 1) - 0.5) < 1e-3:
                note.start += offset
                note.end += offset
    return pm


def shuffle(
    pm: pretty_midi.PrettyMIDI, prob: float, seed: int | None = None
) -> pretty_midi.PrettyMIDI:
    rng = random.Random(seed)
    for inst in pm.instruments:
        inst.notes.sort(key=lambda n: n.start)
        for i in range(len(inst.notes) - 1):
            if rng.random() < prob:
                a = inst.notes[i]
                b = inst.notes[i + 1]
                a.start, b.start = b.start, a.start
                a.end, b.end = b.end, a.end
        inst.notes.sort(key=lambda n: n.start)
    return pm


def transpose(pm: pretty_midi.PrettyMIDI, semitone: int) -> pretty_midi.PrettyMIDI:
    for inst in pm.instruments:
        for note in inst.notes:
            note.pitch += semitone
    return pm


def apply_pipeline(
    pm: pretty_midi.PrettyMIDI,
    *,
    swing_ratio: float | None = None,
    shuffle_prob: float | None = None,
    transpose_amt: int = 0,
) -> pretty_midi.PrettyMIDI:
    if swing_ratio is not None:
        pm = swing(pm, swing_ratio)
    if shuffle_prob is not None and shuffle_prob > 0:
        pm = shuffle(pm, shuffle_prob)
    if transpose_amt:
        pm = transpose(pm, transpose_amt)
    return pm


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - CLI entry
    import argparse

    ap = argparse.ArgumentParser(prog="modcompose augment")
    ap.add_argument("midi", type=Path)
    ap.add_argument("--swing", type=float, default=0.0)
    ap.add_argument("--transpose", type=int, default=0)
    ap.add_argument("--shuffle", type=float, default=0.0)
    ap.add_argument("-o", "--out", type=Path, required=True)
    ns = ap.parse_args(argv)
    pm = pretty_midi.PrettyMIDI(str(ns.midi))
    pm = apply_pipeline(
        pm,
        swing_ratio=ns.swing if ns.swing else None,
        shuffle_prob=ns.shuffle if ns.shuffle else None,
        transpose_amt=ns.transpose,
    )
    pm.write(str(ns.out))
    print(f"wrote {ns.out}")


if __name__ == "__main__":  # pragma: no cover - manual
    main()
