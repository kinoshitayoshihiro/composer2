#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from pathlib import Path
from typing import Any

# 楽器カテゴリパターン（Los Angeles MIDI Dataset用）
TARGETS = {
    "bass": [
        re.compile(r"bass", re.IGNORECASE),
    ],
    "guitar": [
        re.compile(r"guitar", re.IGNORECASE),
    ],
    "strings": [
        re.compile(r"string|violin|viola|cello", re.IGNORECASE),
    ],
}


def load_pickles(pickle_paths: list[Path]) -> list[dict[str, Any]]:
    """Load LAMD pickle files and convert to uniform dict format."""
    meta: list[dict[str, Any]] = []
    for pp in pickle_paths:
        try:
            with open(pp, "rb") as f:
                obj = pickle.load(f)
                # LAMD format: list of [filename, [[key, val], ...]]
                if isinstance(obj, list):
                    for item in obj:
                        if isinstance(item, list) and len(item) >= 2 and isinstance(item[0], str):
                            filename = item[0]
                            # Extract program/instrument from nested list
                            props = item[1] if len(item) > 1 else []
                            patches = []
                            for prop in props:
                                if (
                                    isinstance(prop, list)
                                    and len(prop) >= 2
                                    and prop[0] == "midi_patches"
                                ):
                                    patches = prop[1]
                                    break
                            record = {
                                "filename": filename + ".mid",
                                "patches": patches,
                            }
                            meta.append(record)
        except (OSError, pickle.UnpicklingError) as e:
            print(f"[skip-meta] {pp}: {e}", file=sys.stderr)
    return meta


def match_category(patches: list[int], inst: str) -> bool:
    """Match instrument by MIDI program numbers (GM patches)."""
    # GM program ranges:
    # Bass: 32-39 (Acoustic Bass - Synth Bass 2)
    # Guitar: 24-31 (Acoustic Guitar - Distortion Guitar)
    # Strings: 40-47, 48-51 (Violin, Viola, Cello, etc.)
    if inst == "bass":
        return any(32 <= p <= 39 for p in patches)
    elif inst == "guitar":
        return any(24 <= p <= 31 for p in patches)
    elif inst == "strings":
        return any((40 <= p <= 47) or (48 <= p <= 51) for p in patches)
    return False


def discover_meta(root: Path | str) -> list[Path]:
    """指定ディレクトリから .pkl / .pickle メタデータファイルを再帰的に探す"""
    root = Path(root)
    pickles = list(root.rglob("*.pkl"))
    pickles.extend(root.rglob("*.pickle"))
    return pickles


def discover_midis(root: Path | str) -> list[Path]:
    """指定ディレクトリから .mid/.midi ファイルを再帰的に探す"""
    root = Path(root)
    midis = list(root.rglob("*.mid"))
    midis.extend(root.rglob("*.midi"))
    return midis


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "root",
        type=str,
        help="LAMD ルート（展開済みディレクトリ）",
    )
    ap.add_argument(
        "--instrument",
        required=True,
        choices=["bass", "guitar", "strings"],
    )
    ap.add_argument("--out", required=True, help="JSONL 出力パス")
    args = ap.parse_args()

    root = Path(args.root)
    meta_pickles = discover_meta(root)
    midi_files = discover_midis(root)
    midi_index = {p.name: p for p in midi_files}

    metas = load_pickles(meta_pickles)

    n_total, n_out = 0, 0
    with open(args.out, "w", encoding="utf-8") as w:
        for md in metas:
            try:
                fn = str(md.get("filename", ""))
                patches = md.get("patches", [])
                if not isinstance(patches, list):
                    patches = []
                if not fn:
                    continue
                n_total += 1
                if not match_category(patches, args.instrument):
                    continue
                p = midi_index.get(Path(fn).name)
                if not p:
                    # ファイル名マッチに失敗したら、緩めに探す
                    cand = list(root.rglob(Path(fn).name))
                    p = cand[0] if cand else None
                if not p:
                    continue
                rec: dict[str, Any] = {
                    "id": p.stem,
                    "path": str(p.resolve()),
                    "instrument": args.instrument,
                    "source": "LAMD",
                    "meta": {"patches": patches},
                }
                w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_out += 1
            except (OSError, ValueError, KeyError) as e:
                print(f"[skip] {e}", file=sys.stderr)
    print(f"[done] total_meta={n_total} -> matched={n_out}")


if __name__ == "__main__":
    main()
