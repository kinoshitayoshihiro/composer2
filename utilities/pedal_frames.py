from __future__ import annotations

from pathlib import Path
from typing import Iterable

from pathlib import Path

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
    src: Path | pretty_midi.PrettyMIDI,
    *,
    sr: int = SR,
    hop_length: int = HOP_LENGTH,
    cc_num: int = 64,
    cc_th: int = 64,
    max_seconds: float = 0.0,
) -> pd.DataFrame:
    """Return pedal frame features from a MIDI file or PrettyMIDI object."""
    pm: pretty_midi.PrettyMIDI
    if isinstance(src, pretty_midi.PrettyMIDI):
        pm = src
        file_id = "pm"
    else:
        pm = pretty_midi.PrettyMIDI(str(src))
        # caller will override to root-relative; fallback is basename to preserve current behavior
        file_id = Path(src).name

    # Render audio; skip files that produce empty audio to avoid librosa errors
    audio = _get_audio(pm, sr)
    if max_seconds and max_seconds > 0 and audio.size > 0:
        max_len = int(sr * float(max_seconds))
        if max_len < audio.shape[-1]:
            audio = audio[:max_len]
    if audio.size == 0:
        return pd.DataFrame()
    chroma = librosa.feature.chroma_cqt(y=audio, sr=sr, hop_length=hop_length)
    frame_times = librosa.frames_to_time(
        np.arange(chroma.shape[1]), sr=sr, hop_length=hop_length
    )

    pedal_events = [
        cc for inst in pm.instruments for cc in inst.control_changes if cc.number == cc_num
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
            pedal_state = 1 if val >= cc_th else 0

            row = {
                "file": file_id,
                "track_id": track_id,
                "frame_id": frame_id,
                "time_sec": float(t),
                **{f"chroma_{i}": float(chroma[i, frame_id]) for i in range(12)},
                "rel_release": rel_release,
                "pedal_state": pedal_state,
            }
            rows.append(row)  # caller may replace "file" with relpath

    return pd.DataFrame(rows)


def main(argv: Iterable[str] | None = None) -> None:
    import argparse

    p = argparse.ArgumentParser(description="Extract pedal frames from MIDI directory")
    p.add_argument("midi_dir", type=Path)
    p.add_argument("--csv", dest="csv_out", type=Path, required=True)
    p.add_argument("--sr", type=int, default=SR, help=f"sample rate (default {SR})")
    p.add_argument(
        "--hop", type=int, default=HOP_LENGTH, help=f"hop length (default {HOP_LENGTH})"
    )
    p.add_argument("--cc-num", type=int, default=64, help="CC number to treat as pedal (default 64)")
    p.add_argument("--cc-th", type=int, default=64, help="threshold for CC64 on/off (default 64)")
    # workload control / logging
    p.add_argument("--file-list", type=Path, help="optional text file with MIDI paths to process (one per line)")
    p.add_argument("--fail-log", type=Path, help="append paths that failed to process here")
    p.add_argument("--success-log", type=Path, help="append paths that processed successfully here")
    p.add_argument("--shards", type=int, default=1, help="split workload into N shards")
    p.add_argument("--shard-index", type=int, default=0, help="which shard to process [0..shards-1]")
    p.add_argument("--skip-files", type=int, default=0, help="skip first N files after sharding")
    p.add_argument("--limit-files", type=int, default=0, help="process at most N files (0=all)")
    p.add_argument("--max-seconds", type=float, default=0.0, help="limit audio render to first N seconds (0=full)")
    args = p.parse_args(list(argv) if argv else None)

    # Resolve file list
    midi_paths: list[Path]
    if args.file_list and args.file_list.is_file():
        midi_paths = [Path(line.strip()) for line in args.file_list.read_text().splitlines() if line.strip()]
    else:
        midi_paths = sorted(args.midi_dir.rglob("*.mid")) + sorted(args.midi_dir.rglob("*.midi"))
    # shard selection
    shards = max(1, int(args.shards))
    shard_idx = max(0, int(args.shard_index)) % shards
    if shards > 1:
        midi_paths = [p for i, p in enumerate(midi_paths) if i % shards == shard_idx]
    # skip/limit
    if args.skip_files > 0:
        midi_paths = midi_paths[args.skip_files :]
    if args.limit_files and args.limit_files > 0:
        midi_paths = midi_paths[: args.limit_files]
    if not midi_paths:
        print("no MIDI files found")
        return
    # Stream write to avoid memory blow-up
    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    header_written = False
    total_rows = 0
    total_pos = 0
    pos_files = 0
    processed = 0
    # open logs in append mode if requested
    fail_f = None
    succ_f = None
    try:
        if args.fail_log:
            args.fail_log.parent.mkdir(parents=True, exist_ok=True)
            fail_f = args.fail_log.open("a")
        if args.success_log:
            args.success_log.parent.mkdir(parents=True, exist_ok=True)
            succ_f = args.success_log.open("a")
        for path in midi_paths:
            if any(seg.startswith('.') for seg in path.parts):
                continue
            try:
                df = extract_from_midi(
                    path,
                    sr=args.sr,
                    hop_length=args.hop,
                    cc_num=args.cc_num,
                    cc_th=args.cc_th,
                    max_seconds=float(args.max_seconds or 0.0),
                )
                # ensure file id is unique: use path relative to root when available
                if not df.empty:
                    try:
                        if args.midi_dir:
                            rel = path.relative_to(args.midi_dir)
                            df["file"] = rel.as_posix()
                        else:
                            df["file"] = path.name
                    except Exception:
                        df["file"] = path.name
                    mode = "a" if header_written else "w"
                    df.to_csv(args.csv_out, index=False, mode=mode, header=not header_written)
                    header_written = True
                    total_rows += len(df)
                    pos_frames = int(df["pedal_state"].sum()) if "pedal_state" in df else 0
                    total_pos += pos_frames
                    if pos_frames > 0:
                        pos_files += 1
                # success (even if empty)
                if succ_f:
                    succ_f.write(str(path) + "\n")
            except Exception as e:
                if fail_f:
                    fail_f.write(str(path) + "\n")
            processed += 1
            if processed % 100 == 0:
                print(f"processed {processed}/{len(midi_paths)} files... rows={total_rows} positives={total_pos}")
    finally:
        if fail_f:
            fail_f.close()
        if succ_f:
            succ_f.close()
    print(f"wrote {total_rows} rows to {args.csv_out}")
    print(f"scanned={len(midi_paths)} positives={pos_files} frame_positives={total_pos}")


if __name__ == "__main__":  # pragma: no cover
    main()
