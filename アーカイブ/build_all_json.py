#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import json
from music21 import converter, note, chord, meter


def build_heatmap(midi_path, resolution):
    # （先ほどの onset_heatmap.py と同じ内容。略。）
    score = converter.parse(midi_path)
    ts = score.recurse().getElementsByClass(meter.TimeSignature)
    if len(ts) == 0:
        raise RuntimeError("拍子情報が見つかりませんでした。")
    first_ts = ts[0]
    beats_per_measure = first_ts.numerator
    beat_unit = first_ts.denominator
    quarter_per_measure = beats_per_measure * (4.0 / beat_unit)

    onset_offsets = []
    for el in score.recurse():
        if isinstance(el, note.Note) or isinstance(el, chord.Chord):
            onset_offsets.append(el.offset)
    if len(onset_offsets) == 0:
        raise RuntimeError("ノート（Note/Chord）のオンセットが見つかりませんでした。")

    max_offset = max(onset_offsets)
    total_measures = int(max_offset // quarter_per_measure) + 1
    total_grids = total_measures * resolution
    heatmap_counts = [0] * total_grids

    for off in onset_offsets:
        measure_index = int(off // quarter_per_measure)
        offset_in_measure = off - (measure_index * quarter_per_measure)
        subbeat_index = int((offset_in_measure / quarter_per_measure) * resolution)
        if subbeat_index >= resolution:
            subbeat_index = resolution - 1
        grid_index = measure_index * resolution + subbeat_index
        if 0 <= grid_index < total_grids:
            heatmap_counts[grid_index] += 1

    heatmap_list = []
    for idx, cnt in enumerate(heatmap_counts):
        heatmap_list.append({"grid_index": idx, "count": cnt})

    return heatmap_list


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
