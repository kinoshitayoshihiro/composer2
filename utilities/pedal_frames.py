from __future__ import annotations

from pathlib import Path
from typing import Iterable

import librosa
import numpy as np
import pandas as pd
import pretty_midi

from .time_utils import get_end_time

SR = 22050
HOP_LENGTH = 512


def _get_audio(pm: pretty_midi.PrettyMIDI, sr: int = SR) -> np.ndarray:
    try:
        return pm.fluidsynth(fs=sr)
    except Exception:  # pragma: no cover - fluidsynth optional
        return pm.synthesize(fs=sr)


def extract_from_midi(
    src: Path | pretty_midi.PrettyMIDI, *, sr: int = SR, hop_length: int = HOP_LENGTH
) -> pd.DataFrame:
    """Return pedal frame features from a MIDI file or PrettyMIDI object."""
    pm = (
        src
        if isinstance(src, pretty_midi.PrettyMIDI)
        else pretty_midi.PrettyMIDI(str(src))
    )

    audio = _get_audio(pm, sr)
    chroma = librosa.feature.chroma_cqt(y=audio, sr=sr, hop_length=hop_length)
    frame_times = librosa.frames_to_time(
        np.arange(chroma.shape[1]), sr=sr, hop_length=hop_length
    )

    pedal_events = [
        cc for inst in pm.instruments for cc in inst.control_changes if cc.number == 64
    ]
    pedal_events.sort(key=lambda c: c.time)
    pedal_times = np.array([cc.time for cc in pedal_events])
    pedal_vals = np.array([cc.value for cc in pedal_events])

    release_times_by_track: list[np.ndarray] = []
    for inst in pm.instruments:
        release_times = np.array([note.end for note in inst.notes])
        release_times.sort()
        release_times_by_track.append(release_times)

    rows = []
    for track_id, rel_times in enumerate(release_times_by_track):
        for frame_id, t in enumerate(frame_times):
            idx = np.searchsorted(rel_times, t)
            next_rel = rel_times[idx] if idx < len(rel_times) else get_end_time(pm)
            rel_release = float(next_rel - t)

            pidx = np.searchsorted(pedal_times, t, side="right") - 1
            val = pedal_vals[pidx] if pidx >= 0 else 0
            pedal_state = 1 if val >= 64 else 0

            row = {
                "track_id": track_id,
                "frame_id": frame_id,
                **{f"chroma_{i}": float(chroma[i, frame_id]) for i in range(12)},
                "rel_release": rel_release,
                "pedal_state": pedal_state,
            }
            rows.append(row)

    return pd.DataFrame(rows)


def main(argv: Iterable[str] | None = None) -> None:
    import argparse

    p = argparse.ArgumentParser(description="Extract pedal frames from MIDI directory")
    p.add_argument("midi_dir", type=Path)
    p.add_argument("--csv", dest="csv_out", type=Path, required=True)
    args = p.parse_args(list(argv) if argv else None)

    all_frames = []
    for path in sorted(args.midi_dir.glob("*.mid")):
        df = extract_from_midi(path)
        all_frames.append(df)

    if all_frames:
        result = pd.concat(all_frames, ignore_index=True)
        result.to_csv(args.csv_out, index=False)
        print(f"wrote {len(result)} rows to {args.csv_out}")
    else:
        print("no MIDI files found")


if __name__ == "__main__":  # pragma: no cover
    main()
