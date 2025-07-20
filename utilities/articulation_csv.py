from __future__ import annotations

from dataclasses import asdict, dataclass
from bisect import bisect_right
from pathlib import Path
from typing import Iterable, List
import argparse
import pandas as pd

try:
    import pretty_midi
except Exception:  # pragma: no cover - optional
    pretty_midi = None  # type: ignore


# NOTE: this module is a skeleton for later expansion. Articulation
# extraction will evolve, but for now we simply gather basic note
# information from a MIDI file and include sustain pedal state.


@dataclass(slots=True)
class ArticRow:
    track_id: int
    onset: float
    duration: float
    velocity: int
    pedal_state: int
    articulation_label: str


def extract_from_midi(midi_path: Path) -> List[ArticRow]:
    """Return articulation rows extracted from ``midi_path``."""
    if pretty_midi is None:  # pragma: no cover - optional dependency
        raise RuntimeError("pretty_midi required")

    pm = pretty_midi.PrettyMIDI(str(midi_path))

    pedal_events: list[tuple[float, int]] = []
    for inst in pm.instruments:
        for cc in inst.control_changes:
            if cc.number == 64:
                state = 1 if cc.value >= 64 else 2 if cc.value >= 40 else 0
                pedal_events.append((cc.time, state))
    pedal_events.sort(key=lambda x: x[0])
    pedal_times = [t for t, _ in pedal_events]
    pedal_states = [s for _, s in pedal_events]

    def pedal_state_at(time_sec: float) -> int:
        idx = bisect_right(pedal_times, time_sec) - 1
        return pedal_states[idx] if idx >= 0 else 0

    rows: list[ArticRow] = []
    for track_id, inst in enumerate(pm.instruments):
        for note in inst.notes:
            rows.append(
                ArticRow(
                    track_id=track_id,
                    onset=note.start,
                    duration=note.end - note.start,
                    velocity=note.velocity,
                    pedal_state=pedal_state_at(note.start),
                    articulation_label="",
                )
            )

    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Extract articulation features")  # pragma: no cover - CLI setup
    ap.add_argument("midi", type=Path)  # pragma: no cover - CLI setup
    ap.add_argument("--out", type=Path, required=True)  # pragma: no cover - CLI setup
    ap.add_argument("--only-pedal", action="store_true")  # pragma: no cover - CLI setup
    ns = ap.parse_args(argv)  # pragma: no cover - CLI setup

    rows = extract_from_midi(ns.midi)  # pragma: no cover - CLI execution
    if ns.only_pedal:  # pragma: no cover - CLI execution
        pm = pretty_midi.PrettyMIDI(str(ns.midi))  # pragma: no cover - CLI execution
        data = []  # pragma: no cover - CLI execution
        pedal_iter = iter(rows)  # pragma: no cover - CLI execution
        for track_id, inst in enumerate(pm.instruments):  # pragma: no cover - CLI execution
            for note in inst.notes:  # pragma: no cover - CLI execution
                pedal_state = next(pedal_iter).pedal_state  # pragma: no cover - CLI execution
                data.append(
                    {
                        "track_id": track_id,
                        "onset": note.start,
                        "pitch": note.pitch,
                        "pedal_state": pedal_state,
                    }
                )  # pragma: no cover - CLI execution
        pd.DataFrame(data).to_csv(ns.out, index=False)  # pragma: no cover - CLI execution
    else:
        df = pd.DataFrame([asdict(r) for r in rows])  # pragma: no cover - CLI execution
        df.to_csv(ns.out, index=False)  # pragma: no cover - CLI execution
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())


__all__ = ["ArticRow", "extract_from_midi", "main"]
