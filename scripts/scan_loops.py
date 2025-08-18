import argparse
import csv
import logging
import re
from pathlib import Path

import pretty_midi

from utilities.groove_sampler_v2 import _safe_read_bpm, mido
from utilities.pretty_midi_safe import pm_to_mido

if mido is None:  # pragma: no cover - dependency is required
    raise RuntimeError("mido is required for loop scanning; install via 'pip install mido'")

logger = logging.getLogger(__name__)


def _analyze(path: Path):
    logger.info("Analyzing %s", path)
    pm = pretty_midi.PrettyMIDI(str(path))
    bpm = _safe_read_bpm(pm, default_bpm=120.0, fold_halves=False)
    if getattr(_safe_read_bpm, "last_source", "") == "default":
        logger.warning("Tempo not found in %s; using default 120 BPM", path)
    midi = pm_to_mido(pm)
    ticks_per_beat = midi.ticks_per_beat
    total_ticks = max(sum(msg.time for msg in tr) for tr in midi.tracks)
    ts_msg = None
    for tr in midi.tracks:
        for msg in tr:
            if msg.type == "time_signature":
                ts_msg = msg
                break
        if ts_msg:
            break
    beats_per_bar = ts_msg.numerator * (4 / ts_msg.denominator) if ts_msg else 4.0
    bars = total_ticks / (ticks_per_beat * beats_per_bar)
    all_notes = [n for inst in pm.instruments for n in inst.notes]
    notes = len(all_notes)
    uniq = len({n.pitch for n in all_notes})
    is_drum = any(inst.is_drum for inst in pm.instruments)
    is_fill = bool(re.search(r"\bfill\b", path.name, re.IGNORECASE))
    keep_drum = is_drum
    keep_pitched = not is_drum
    return [
        str(path),
        path.stat().st_size,
        f"{bpm:.2f}",
        f"{bars:.2f}",
        notes,
        uniq,
        int(is_drum),
        int(is_fill),
        int(keep_drum),
        int(keep_pitched),
    ]


def main():
    ap = argparse.ArgumentParser(description="Scan loops and write inventory CSV")
    ap.add_argument("--root", type=Path, default=Path("data/loops"))
    ap.add_argument("--out", type=Path, default=Path("loop_inventory.csv"))
    ns = ap.parse_args()

    rows = []
    for p in ns.root.rglob("*.mid"):
        try:
            rows.append(_analyze(p))
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to analyze %s: %s", p, exc)
    with ns.out.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "path",
                "bytes",
                "bpm",
                "bars",
                "notes",
                "uniq_pitches",
                "drum",
                "is_fill",
                "keep_drum",
                "keep_pitched",
            ]
        )
        writer.writerows(rows)


if __name__ == "__main__":  # pragma: no cover
    main()
