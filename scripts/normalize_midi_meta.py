from __future__ import annotations

"""Normalize MIDI meta events by moving global meta (tempo, key, time signature)
to track 0. This helps libraries like pretty_midi which expect such meta events
on track 0 only.

Usage:
  python -m scripts.normalize_midi_meta --in data/songs --out-dir data/songs_norm
  python -m scripts.normalize_midi_meta --in some/file.mid --out-dir out_midis

Notes:
  - Only the following meta events are moved to track 0:
      set_tempo, time_signature, key_signature
  - Relative timing in all tracks is preserved.
  - Output is written under --out-dir, preserving relative paths.
"""

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import mido


META_TYPES = {"set_tempo", "time_signature", "key_signature"}


def _collect_global_meta(mid: mido.MidiFile) -> list[tuple[int, mido.MetaMessage]]:
    """Collect global meta messages from all tracks with absolute tick times."""
    events: list[tuple[int, mido.MetaMessage]] = []
    for ti, track in enumerate(mid.tracks):
        t_abs = 0
        for msg in track:
            t_abs += msg.time
            if getattr(msg, "is_meta", False) and msg.type in META_TYPES:
                # copy message to avoid mutating original
                events.append((t_abs, msg.copy()))
    # Sort by absolute time to reconstruct in track 0
    events.sort(key=lambda x: x[0])
    # Deduplicate identical meta at same time to avoid duplicates
    deduped: list[tuple[int, mido.MetaMessage]] = []
    seen: set[tuple] = set()
    for t_abs, msg in events:
        key: tuple
        if msg.type == "set_tempo":
            key = (t_abs, msg.type, int(msg.tempo))
        elif msg.type == "time_signature":
            key = (
                t_abs,
                msg.type,
                int(msg.numerator),
                int(msg.denominator),
                int(getattr(msg, "clocks_per_click", 24)),
                int(getattr(msg, "notated_32nd_notes_per_beat", 8)),
            )
        elif msg.type == "key_signature":
            key = (t_abs, msg.type, str(msg.key))
        else:
            key = (t_abs, msg.type)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((t_abs, msg))
    return deduped


def _rewrite_tracks_without_global_meta(tracks: Iterable[mido.MidiTrack]) -> list[mido.MidiTrack]:
    """Return new tracks where global meta events are removed,
    preserving timing by carrying their delta time to the next kept message.
    """
    new_tracks: list[mido.MidiTrack] = []
    for track in tracks:
        new_tr = mido.MidiTrack()
        carry = 0
        for msg in track:
            if getattr(msg, "is_meta", False) and msg.type in META_TYPES:
                carry += msg.time
                continue
            # Keep message; its time receives any carried delta
            new_tr.append(msg.copy(time=msg.time + carry))
            carry = 0
        new_tracks.append(new_tr)
    return new_tracks


def normalize_file(in_path: Path, out_path: Path) -> tuple[int, int]:
    """Normalize one MIDI file and write to out_path.

    Returns (moved_count, track_count)
    """
    mid = mido.MidiFile(filename=str(in_path))
    meta_events = _collect_global_meta(mid)

    # Rebuild track 0 with the collected meta messages
    tr0 = mido.MidiTrack()
    tr0.append(mido.MetaMessage("track_name", name="meta", time=0))
    prev = 0
    for t_abs, msg in meta_events:
        delta = max(0, int(t_abs) - int(prev))
        tr0.append(msg.copy(time=delta))
        prev = t_abs
    tr0.append(mido.MetaMessage("end_of_track", time=0))

    # Rebuild other tracks without those global meta messages
    other_tracks = _rewrite_tracks_without_global_meta(mid.tracks)
    # Ensure our new track 0 is first
    new_mid = mido.MidiFile(ticks_per_beat=mid.ticks_per_beat, type=mid.type)
    new_mid.tracks.append(tr0)
    # Append the rest; if original had N tracks, we keep N tracks (replacing 0th)
    for i, tr in enumerate(other_tracks):
        if i == 0:
            continue  # replaced by our meta track
        new_mid.tracks.append(tr)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    new_mid.save(str(out_path))
    return len(meta_events), len(new_mid.tracks)


def find_mid_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return [p for p in root.rglob("*") if p.suffix.lower() in {".mid", ".midi"}]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", type=Path, required=True, help="MIDI file or directory")
    ap.add_argument("--out-dir", type=Path, required=True, help="output directory for normalized MIDIs")
    ap.add_argument("--dry-run", action="store_true", help="scan and report only; no files written")
    args = ap.parse_args(argv)

    src = args.inp
    out_root = args.out_dir
    files = find_mid_files(src)
    if not files:
        print("No MIDI files found.")
        return 1

    moved_total = 0
    for f in files:
        rel = f.name if src.is_file() else f.relative_to(src)
        out = out_root / rel
        if args.dry_run:
            try:
                mid = mido.MidiFile(filename=str(f))
                moved = len(_collect_global_meta(mid))
                print(f"SCAN {f} -> {moved} global meta events")
                moved_total += moved
            except Exception as e:
                print(f"ERROR reading {f}: {e}")
            continue
        try:
            moved, ntracks = normalize_file(f, out)
            moved_total += moved
            print(f"WROTE {out} (moved {moved} meta, tracks {ntracks})")
        except Exception as e:
            print(f"ERROR processing {f}: {e}")
    print(f"Done. Files: {len(files)}, total moved meta: {moved_total}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

