#!/usr/bin/env python3
from __future__ import annotations
import argparse, yaml
from typing import List, Tuple
import music21 as m21


def load_score(midi_paths: List[str]) -> m21.stream.Score:
    parts = []
    for p in midi_paths:
        s = m21.converter.parse(p)
        parts.append(s.flatten().parts[0] if s.parts else s.flatten())
    sc = m21.stream.Score()
    for p in parts:
        sc.insert(0, p)
    return sc


def chordify_by_measures(sc: m21.stream.Score, qpm: float | None = None):
    # 小節単位で chordify → 代表和音を選ぶ（多数決＋持続時間）
    s = sc.chordify()
    # キー推定（Krumhansl）
    key = s.analyze("Krumhansl")
    out = []
    for meas in s.makeMeasures():
        chords = [n for n in meas.recurse().getElementsByClass("Chord")]
        if not chords:
            continue
        # 最長持続の和音を代表に（過学習防止に単純ルール）
        rep = max(chords, key=lambda c: c.duration.quarterLength)
        rep.closedPosition(forceOctave=4, inPlace=True)
        # 和音名を安定化
        rep = rep.simplifyEnharmonics()
        rn = m21.roman.romanNumeralFromChord(rep, key)
        out.append(
            {
                "bar": int(meas.measureNumber or 0),
                "root": rep.root().name,  # e.g., D
                "quality": rep.commonName,  # e.g., major-seventh chord
                "roman": rn.figure if rn else None,
                "dur": 1,  # 1小節想定（後でテンポ/拍に合わせて再配分可）
            }
        )
    return str(key), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("midi", nargs="+", help="和声音源のMIDI（gtr/pf/bass推奨）")
    ap.add_argument("--out", default="base_chordmap.yaml")
    args = ap.parse_args()
    sc = load_score(args.midi)
    key, chords = chordify_by_measures(sc)
    data = {"key": str(key), "base_chords": chords}
    with open(args.out, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


if __name__ == "__main__":
    main()
