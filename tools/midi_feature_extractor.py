"""Extract simple features from a MIDI file using pretty_midi.

Usage:
    python tools/midi_feature_extractor.py path/to/file.mid [-o output.json]
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict
import argparse

import pretty_midi


def extract_features(midi_path: str | Path) -> Dict[str, float]:
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    notes = [n for inst in pm.instruments for n in inst.notes]
    total_notes = len(notes)
    mean_velocity = float(sum(n.velocity for n in notes) / total_notes) if notes else 0.0
    tempi, _times = pm.get_tempo_changes()
    tempo = float(sum(tempi) / len(tempi)) if len(tempi) else 0.0
    length = pm.get_end_time() or 1.0
    density = float(total_notes) / length
    return {
        "tempo_bpm": tempo,
        "total_notes": total_notes,
        "mean_velocity": mean_velocity,
        "note_density_per_sec": density,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract MIDI features")
    parser.add_argument("midi_file", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    args = parser.parse_args()
    feats = extract_features(args.midi_file)
    text = json.dumps(feats, indent=2)
    if args.output:
        args.output.write_text(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
