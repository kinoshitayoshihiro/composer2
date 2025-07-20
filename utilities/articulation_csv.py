from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pretty_midi

from .duration_bucket import to_bucket
from .time_utils import seconds_to_qlen


def extract_from_midi(src: Path | pretty_midi.PrettyMIDI) -> pd.DataFrame:
    """Return note features for ML training from a MIDI file or object."""
    pm = (
        src
        if isinstance(src, pretty_midi.PrettyMIDI)
        else pretty_midi.PrettyMIDI(str(src))
    )

    pedal_events = [
        cc for inst in pm.instruments for cc in inst.control_changes if cc.number == 64
    ]
    pedal_events.sort(key=lambda x: x.time)
    pedal_times = np.array([cc.time for cc in pedal_events])
    pedal_vals = np.array([cc.value for cc in pedal_events])

    rows: list[dict[str, float | int | None]] = []
    for tid, inst in enumerate(pm.instruments):
        for note in inst.notes:
            onset = seconds_to_qlen(pm, note.start)
            qlen = seconds_to_qlen(pm, note.end) - onset
            cc_idx = np.searchsorted(pedal_times, note.start, side="right") - 1
            val = pedal_vals[cc_idx] if cc_idx >= 0 else 0
            if val >= 64:
                pedal_state = 1
            elif val >= 40:
                pedal_state = 2
            else:
                pedal_state = 0
            rows.append(
                {
                    "track_id": tid,
                    "pitch": note.pitch,
                    "onset": onset,
                    "duration": qlen,
                    "velocity": note.velocity / 127.0,
                    "pedal_state": pedal_state,
                    "bucket": to_bucket(qlen),
                    "articulation_label": None,
                }
            )

    return pd.DataFrame(rows)


__all__ = ["extract_from_midi"]


def _main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Extract articulation features")
    p.add_argument("midi_dir", type=Path)
    p.add_argument("--csv", type=Path, required=True)
    args = p.parse_args()

    rows = []
    for midi in sorted(args.midi_dir.glob("*.mid")):
        df = extract_from_midi(midi)
        rows.append(df)
    if rows:
        pd.concat(rows).to_csv(args.csv, index=False)


if __name__ == "__main__":  # pragma: no cover - CLI
    _main()
