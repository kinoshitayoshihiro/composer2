from __future__ import annotations

"""Convert frame-wise pedal CSV to span-wise CSV (song_id,start_sec,end_sec,pedal).

Assumes columns: 'file' (or 'song_id'), 'time_sec', 'pedal_state'.
Frames must be contiguous per song and time_sec increasing; if not, the script
will sort by (song_id, time_sec).

Usage:
  python -m scripts.pedal_frames_to_spans --in data/pedal/all.csv --out data/pedal/all_spans.csv
"""

import argparse
import csv
from pathlib import Path


def write_spans(rows, w):
    cur_state = None
    start_t = None
    prev_t = None
    for t, s in rows:
        if cur_state is None:
            cur_state = s
            start_t = t
        elif s != cur_state:
            w.writerow([f"{start_t:.6f}", f"{t:.6f}", int(cur_state)])
            cur_state = s
            start_t = t
        prev_t = t
    if cur_state is not None and start_t is not None:
        # leave end empty if unknown
        w.writerow([f"{start_t:.6f}", "", int(cur_state)])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--out", dest="out", type=Path, required=True)
    args = ap.parse_args(argv)

    with args.inp.open() as f, args.out.open("w", newline="") as g:
        r = csv.DictReader(f)
        cols = r.fieldnames or []
        sid_col = "file" if "file" in cols else ("song_id" if "song_id" in cols else None)
        if sid_col is None or "time_sec" not in cols or "pedal_state" not in cols:
            raise SystemExit("input CSV must include 'file' or 'song_id', and 'time_sec', 'pedal_state'")
        w = csv.writer(g)
        w.writerow(["song_id", "start_sec", "end_sec", "pedal"])
        cur_sid = None
        buf = []
        for row in r:
            sid = row.get(sid_col, "").strip()
            t = float(row.get("time_sec", 0.0) or 0.0)
            s = int(row.get("pedal_state", 0) or 0)
            if sid != cur_sid and buf:
                # sort by time
                buf.sort(key=lambda x: x[0])
                for_start = [x for x in buf]
                # write spans for previous sid
                w.writerow([cur_sid])  # optional: separator row with song_id only
                write_spans(for_start, w)
                buf.clear()
            cur_sid = sid
            buf.append((t, s))
        if buf:
            buf.sort(key=lambda x: x[0])
            w.writerow([cur_sid])
            write_spans(buf, w)
    print({"wrote": str(args.out)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

