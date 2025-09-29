#!/usr/bin/env python3
"""Prepare piano data for DUV training from slakh.rich.csv"""

import pandas as pd
import numpy as np
import pathlib as p
import json
import sys


def main():
    print("Preparing piano data for DUV training...")

    src = "data/slakh.rich.csv"
    if not p.Path(src).exists():
        print(f"Error: {src} not found!")
        sys.exit(1)

    print(f"Loading data from {src}...")
    df = pd.read_csv(src, low_memory=False)
    print(f"Original data: {len(df)} rows")

    # ピアノ: GM 0-7
    df = df.query("0 <= program <= 7").copy()
    print(f"Piano data: {len(df)} rows")

    if len(df) == 0:
        print("No piano data found!")
        sys.exit(1)

    # 曲IDを推定（file か path っぽい列があればそれで）
    key = "file" if "file" in df.columns else ("path" if "path" in df.columns else None)
    if key is None:
        # フォールバック: ざっくり1000行ごとを曲ブロック扱い（乱数シード固定）
        df["song_id"] = np.arange(len(df)) // 1000
        print("Using fallback song grouping (1000 rows per song)")
    else:
        df["song_id"] = df[key].astype("category").cat.codes
        print(f"Using {key} for song grouping")

    songs = df["song_id"].unique()
    print(f"Total songs: {len(songs)}")

    rng = np.random.default_rng(42)
    rng.shuffle(songs)
    n_valid = max(1, int(0.1 * len(songs)))
    valid_ids = set(songs[:n_valid])
    train = df[~df["song_id"].isin(valid_ids)].copy()
    valid = df[df["song_id"].isin(valid_ids)].copy()

    print(f"Train: {len(train)} rows, {len(songs) - n_valid} songs")
    print(f"Valid: {len(valid)} rows, {n_valid} songs")

    # Output directories
    p.Path("data/duv").mkdir(parents=True, exist_ok=True)
    train.to_csv("data/duv/piano.train.csv", index=False)
    valid.to_csv("data/duv/piano.valid.csv", index=False)
    print("Saved train/valid CSV files")

    # 統計（trainのみから）
    drop = {
        "velocity",
        "duration",
        "beat_bin",
        "bar",
        "position",
        "file",
        "path",
        "midi",
        "track",
        "split",
        "onset",
        "offset",
        "start",
        "end",
        "time",
        "id",
        "note",
        "pitch",
        "pitch_midi",
        "song_id",
        "program",
        "channel",
    }
    num_cols = train.select_dtypes(include=["number"]).columns.tolist()
    feat = [c for c in num_cols if c not in drop]

    mean = {c: float(train[c].mean()) for c in feat}
    std = {}
    for c in feat:
        v = float(train[c].std(ddof=0)) if train[c].notna().any() else 1.0
        std[c] = v if v and v >= 1e-8 else 1.0
    stats = {"feat_cols": feat, "mean": mean, "std": std}

    p.Path("checkpoints").mkdir(exist_ok=True)
    stats_path = "checkpoints/piano.duv.stats.json"
    p.Path(stats_path).write_text(json.dumps(stats, ensure_ascii=False, indent=2))

    print(f"Stats saved to {stats_path}")
    print({"train_rows": len(train), "valid_rows": len(valid), "feat_cols": len(feat)})
    print("Feature columns:", feat[:10], "..." if len(feat) > 10 else "")

    print("Data preparation complete!")


if __name__ == "__main__":
    main()
