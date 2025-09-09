from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pretty_midi

from dataclasses import dataclass

from .duration_bucket import to_bucket
from .time_utils import seconds_to_qlen

__all__ = ["ArticRow", "extract_from_midi", "main"]


@dataclass
class ArticRow:
    """Container for a single articulation training example."""

    track_id: int
    pitch: int
    onset: float
    duration: float
    velocity: float
    pedal_state: int
    bucket: int
    articulation_label: str | None


def extract_from_midi(src: Path | pretty_midi.PrettyMIDI) -> pd.DataFrame:
    """Return note features for ML training from a MIDI file or object."""
    # Load PrettyMIDI object if a file path is provided
    pm = src if isinstance(src, pretty_midi.PrettyMIDI) else pretty_midi.PrettyMIDI(str(src))

    # Collect and sort sustain pedal events (CC #64)
    pedal_events = [
        cc for inst in pm.instruments for cc in inst.control_changes if cc.number == 64
    ]
    pedal_events.sort(key=lambda x: x.time)
    pedal_times = np.array([cc.time for cc in pedal_events])
    pedal_vals = np.array([cc.value for cc in pedal_events])

    rows: list[dict[str, float | int | str | None]] = []
    for track_id, inst in enumerate(pm.instruments):
        for note in inst.notes:
            # Compute onset and duration in quarterLength
            onset = seconds_to_qlen(pm, note.start)
            qlen = seconds_to_qlen(pm, note.end) - onset

            # Determine pedal state at note onset
            idx = np.searchsorted(pedal_times, note.start, side="right") - 1
            val = pedal_vals[idx] if idx >= 0 else 0
            if val >= 64:
                pedal_state = 1
            elif val >= 40:
                pedal_state = 2
            else:
                pedal_state = 0

            rows.append(
                {
                    "track_id": track_id,
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


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Extract articulation features from MIDI directory")
    parser.add_argument("midi_dir", type=Path, help="Directory containing .mid files")
    parser.add_argument("--csv", dest="csv_out", type=Path, required=True, help="Output CSV path")
    args = parser.parse_args()

    # Process all MIDI files in the directory
    all_dfs: list[pd.DataFrame] = []
    for midi_file in sorted(args.midi_dir.glob("*.mid")):
        df = extract_from_midi(midi_file)
        all_dfs.append(df)

    if all_dfs:
        result = pd.concat(all_dfs, ignore_index=True)
        result.to_csv(args.csv_out, index=False)
        print(f"Wrote {len(result)} rows to {args.csv_out}")
    else:
        print("No MIDI files found.")


if __name__ == "__main__":  # pragma: no cover
    main()
