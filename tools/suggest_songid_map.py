from __future__ import annotations

"""Suggest a mapping from validation CSV song_id values to MIDI filename stems.

This helps align per-song thresholds computed on validation with MIDI inference.

Outputs a CSV with columns: song_id,midi_stem,score
Only suggestions with similarity >= --threshold are emitted.

Example:
  python -m tools.suggest_songid_map \
    --val-csv data/phrase_csv/gtr_midi_gap10_valid.csv \
    --midi-dir data/songs_norm \
    --out data/phrase_csv/songid_to_midistem.suggest.csv \
    --threshold 0.65 --topk 1
"""

import argparse
import csv
import difflib
import unicodedata
from pathlib import Path


def _normalize(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).casefold()
    # Keep alphanumeric (Unicode) characters only
    return "".join(ch for ch in s if ch.isalnum())


def _read_unique_song_ids(csv_path: Path, col: str = "song_id") -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    with csv_path.open() as f:
        r = csv.DictReader(f)
        if r.fieldnames is None or col not in r.fieldnames:
            raise SystemExit(f"input CSV must have column {col!r}")
        for row in r:
            sid = str(row.get(col, "")).strip()
            if sid and sid not in seen:
                seen.add(sid)
                out.append(sid)
    return out


def _scan_midi_stems(root: Path) -> list[str]:
    stems: set[str] = set()
    for p in root.rglob("*"):
        if p.suffix.lower() in {".mid", ".midi"}:
            stems.add(p.stem)
    return sorted(stems)


def suggest_map(val_ids: list[str], stems: list[str], *, threshold: float, topk: int) -> list[tuple[str, str, float]]:
    norm_stems = [(s, _normalize(s)) for s in stems]
    out: list[tuple[str, str, float]] = []
    for sid in val_ids:
        sid_norm = _normalize(sid)
        # Exact match (normalized)
        for stem, stem_norm in norm_stems:
            if stem_norm == sid_norm:
                out.append((sid, stem, 1.0))
                break
        else:
            # Fuzzy match by difflib ratio
            best: list[tuple[float, str]] = []
            for stem, stem_norm in norm_stems:
                score = difflib.SequenceMatcher(None, sid_norm, stem_norm).ratio()
                best.append((score, stem))
            best.sort(reverse=True)
            for score, stem in best[: max(1, topk)]:
                if score >= threshold:
                    out.append((sid, stem, score))
                    break
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--val-csv", type=Path, required=True)
    ap.add_argument("--midi-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threshold", type=float, default=0.65)
    ap.add_argument("--topk", type=int, default=1)
    args = ap.parse_args(argv)

    val_ids = _read_unique_song_ids(args.val_csv, col="song_id")
    stems = _scan_midi_stems(args.midi_dir)
    pairs = suggest_map(val_ids, stems, threshold=args.threshold, topk=args.topk)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["song_id", "midi_stem", "score"])
        for sid, stem, score in pairs:
            w.writerow([sid, stem, f"{score:.4f}"])
    print(f"wrote {args.out} suggestions={len(pairs)} (threshold>={args.threshold})")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

