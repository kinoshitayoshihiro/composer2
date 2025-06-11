#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import json
from music21 import converter, meter
from utilities.onset_heatmap import build_heatmap


def extract_tempo_map(midi_path):
    # （先ほどの extract_tempo_map.py と同じ内容。略。）
    score = converter.parse(midi_path)
    tempo_map = []
    for startOffset, endOffset, mm in score.metronomeMarkBoundaries():
        tempo_map.append({"offset_q": float(startOffset), "bpm": float(mm.number)})
    return tempo_map


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("使い方: python3 build_all_json.py <vocal_midi_path> <resolution>")
        print("例) python3 build_all_json.py data/vocal_ore.midi 16")
        sys.exit(1)

    midi_path = sys.argv[1]
    resolution = int(sys.argv[2])

    # 1) ヒートマップを生成
    try:
        heatmap = build_heatmap(midi_path, resolution)
    except Exception as e:
        print(f"[ERROR] Heatmap 生成失敗: {e}")
        sys.exit(1)

    with open("heatmap.json", "w", encoding="utf-8") as f:
        json.dump(heatmap, f, ensure_ascii=False, indent=2)
    print("[OK] heatmap.json を出力しました。")

    # 2) テンポマップを生成
    try:
        tempo_map = extract_tempo_map(midi_path)
    except Exception as e:
        print(f"[ERROR] Tempo Map 抽出失敗: {e}")
        sys.exit(1)

    with open("tempo_map.json", "w", encoding="utf-8") as f:
        json.dump(tempo_map, f, ensure_ascii=False, indent=2)
    print("[OK] tempo_map.json を出力しました。")
