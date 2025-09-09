from __future__ import annotations

"""Create a symlinked MIDI directory where filenames (stems) equal validation CSV song_id.

This allows per-song thresholds computed on validation (keyed by song_id) to apply
directly to MIDI inference, since predict_phrase uses the MIDI stem as song_id.

Example:
  python -m scripts.symlink_midis_by_map \
    --src data/songs_norm \
    --dst data/songs_by_songid \
    --map data/phrase_csv/songid_to_midistem.csv \
    --map-key song_id --map-to midi_stem
"""

import argparse
from pathlib import Path
import sys


def find_midi(root: Path, stem: str) -> Path | None:
    stem_lc = stem.lower()
    for p in root.rglob("*"):
        if p.suffix.lower() in {".mid", ".midi"} and p.stem.lower() == stem_lc:
            return p
    return None


def read_map(path: Path, key: str, to: str) -> list[tuple[str, str]]:
    import csv

    with path.open() as f:
        r = csv.DictReader(f)
        if r.fieldnames is None or key not in r.fieldnames or to not in r.fieldnames:
            raise SystemExit(f"mapping CSV must have columns {key!r} and {to!r}")
        out: list[tuple[str, str]] = []
        for row in r:
            k = str(row.get(key, "")).strip()
            v = str(row.get(to, "")).strip()
            if k and v:
                out.append((k, v))
        return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, required=True, help="source MIDI directory")
    ap.add_argument("--dst", type=Path, required=True, help="destination directory for symlinks")
    ap.add_argument("--map", type=Path, required=True, help="CSV mapping song_id -> midi_stem")
    ap.add_argument("--map-key", type=str, default="song_id")
    ap.add_argument("--map-to", type=str, default="midi_stem")
    args = ap.parse_args(argv)

    pairs = read_map(args.map, args.map_key, args.map_to)
    args.dst.mkdir(parents=True, exist_ok=True)
    ok = 0
    miss = 0
    for sid, stem in pairs:
        p = find_midi(args.src, stem)
        if not p:
            print(f"WARN: midi not found for stem={stem}", file=sys.stderr)
            miss += 1
            continue
        # link name uses song_id
        out = args.dst / f"{sid}{p.suffix.lower()}"
        if out.exists() or out.is_symlink():
            out.unlink()
        out.symlink_to(p.resolve())
        ok += 1
    print(f"created {ok} links; missing {miss}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

