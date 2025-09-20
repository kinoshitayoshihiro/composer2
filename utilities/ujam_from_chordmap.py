#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""UJAM Sparkle向けのMIDI生成ラッパー。

chordmap.final.yaml と tempo_map.json を読み込み、Sparkleの Chord / Common
トラックで利用できる2種類のMIDIファイルを出力する。
既存ユーティリティを変更せず、追加スクリプトとして提供する。
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Iterable, List, Sequence

import pretty_midi as pm
import yaml


DEFAULT_MAPPING: Dict[str, Any] = {
    "chord_octave": 4,
    "chord_len_guard_s": 0.05,
    "pulse_note": 36,
    "pulse_subdiv": 4,
    "pulse_len_s": 0.04,
}

NOTE_NAME_TO_PC = {
    "C": 0,
    "B#": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "Fb": 4,
    "F": 5,
    "E#": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
    "Cb": 11,
}


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def ensure_dir(path: str) -> None:
    if path and not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def pitch_class(root: str) -> int:
    key = root.strip()
    if key in NOTE_NAME_TO_PC:
        return NOTE_NAME_TO_PC[key]
    letters = "".join(ch for ch in key if not ch.isdigit())
    if letters in NOTE_NAME_TO_PC:
        return NOTE_NAME_TO_PC[letters]
    raise ValueError(f"Unknown root name: {root}")


def triad_intervals(quality: str) -> Sequence[int]:
    text = (quality or "").lower()
    if "dim" in text or "°" in text:
        return (0, 3, 6)
    if "aug" in text or "+" in text:
        return (0, 4, 8)
    if "m" in text and "maj" not in text:
        return (0, 3, 7)
    return (0, 4, 7)


def bar_window(downbeats: Sequence[float], bar: int, dur: int) -> tuple[float, float]:
    start_index = max(0, bar - 1)
    end_index = min(len(downbeats) - 1, start_index + dur)
    start = downbeats[start_index]
    if end_index > start_index:
        end = downbeats[end_index]
    else:
        end = start + 1.0
    return start, end


def build_chord_instrument(
    chords: Iterable[Dict[str, Any]],
    downbeats: Sequence[float],
    mapping: Dict[str, Any],
) -> pm.Instrument:
    octave = int(mapping.get("chord_octave", DEFAULT_MAPPING["chord_octave"]))
    guard = float(mapping.get("chord_len_guard_s", DEFAULT_MAPPING["chord_len_guard_s"]))
    inst = pm.Instrument(program=0, is_drum=False, name="UJAM_Sparkle_Chords")
    for record in chords:
        bar = int(record.get("bar", 1))
        dur = int(record.get("dur", 1))
        root = str(record.get("root", "C"))
        quality = str(record.get("quality", ""))
        start, end = bar_window(downbeats, bar, dur)
        end = max(start, end - guard)
        root_pitch = 12 * octave + pitch_class(root)
        for interval in triad_intervals(quality):
            inst.notes.append(
                pm.Note(
                    velocity=96,
                    pitch=root_pitch + interval,
                    start=start,
                    end=end,
                )
            )
    return inst


def select_pulse_times(
    beats: Sequence[float],
    start: float,
    end: float,
    subdiv: int,
) -> List[float]:
    inside = [t for t in beats if start <= t < end]
    if not inside:
        return []
    if subdiv <= 1 or len(inside) <= 1:
        return [inside[0]]
    step = max(1, len(inside) // subdiv)
    chosen = []
    added = set()
    for idx in range(0, len(inside), step):
        idx = min(idx, len(inside) - 1)
        if idx not in added:
            chosen.append(inside[idx])
            added.add(idx)
    if (len(inside) - 1) not in added:
        chosen.append(inside[-1])
    return chosen


def build_pulse_instrument(
    downbeats: Sequence[float],
    beats: Sequence[float],
    mapping: Dict[str, Any],
) -> pm.Instrument:
    note_num = int(mapping.get("pulse_note", DEFAULT_MAPPING["pulse_note"]))
    subdiv = int(mapping.get("pulse_subdiv", DEFAULT_MAPPING["pulse_subdiv"]))
    length = float(mapping.get("pulse_len_s", DEFAULT_MAPPING["pulse_len_s"]))
    inst = pm.Instrument(program=0, is_drum=True, name="UJAM_Sparkle_Pulse")
    for i in range(len(downbeats) - 1):
        start = downbeats[i]
        end = downbeats[i + 1]
        for t in select_pulse_times(beats, start, end, subdiv):
            inst.notes.append(
                pm.Note(
                    velocity=100,
                    pitch=note_num,
                    start=t,
                    end=min(t + length, end),
                )
            )
    return inst


def parse_meter(value: Any) -> int:
    if isinstance(value, str):
        head = value.split("/")[0]
        return int(head)
    if isinstance(value, (int, float)):
        return int(value)
    return 4


def main() -> None:
    parser = argparse.ArgumentParser(description="Write UJAM Sparkle MIDI from chordmap + tempo map")
    parser.add_argument("--chordmap", required=True)
    parser.add_argument("--tempo-map", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--map", default=None, help="optional YAML for mapping/parameters")
    args = parser.parse_args()

    chordmap = load_yaml(args.chordmap)
    tempo_map = load_json(args.tempo_map)
    mapping = dict(DEFAULT_MAPPING)
    if args.map and os.path.exists(args.map):
        mapping.update(load_yaml(args.map))

    final = chordmap.get("final_chords") or chordmap.get("base_chords")
    if not final:
        raise SystemExit("ERROR: chordmap に final_chords/base_chords がありません。")
    downbeats = tempo_map.get("downbeats") or []
    beats = tempo_map.get("beats") or []
    if not downbeats:
        raise SystemExit("ERROR: tempo_map に downbeats が必要です。")

    ensure_dir(args.out_dir)

    chord_inst = build_chord_instrument(final, downbeats, mapping)
    pulse_inst = build_pulse_instrument(downbeats, beats or downbeats, mapping)

    midi_chords = pm.PrettyMIDI()
    midi_chords.instruments.append(chord_inst)
    midi_pulse = pm.PrettyMIDI()
    midi_pulse.instruments.append(pulse_inst)

    chords_path = os.path.join(args.out_dir, "sparkle_chords.mid")
    pulse_path = os.path.join(args.out_dir, "sparkle_pulse.mid")
    midi_chords.write(chords_path)
    midi_pulse.write(pulse_path)
    print(f"Wrote {chords_path} and {pulse_path}")


if __name__ == "__main__":
    main()
