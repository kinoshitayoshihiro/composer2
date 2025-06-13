"""Utilities for creating onset heatmaps from MIDI files."""

from __future__ import annotations

import json
from typing import List
from pathlib import Path

from music21 import converter, note, chord, meter

RESOLUTION = 16  # number of grid bins per measure


from typing import Dict


def load_heatmap(json_path: str) -> Dict[int, int]:
    """Load onset heatmap from a JSON file.

    Parameters
    ----------
    json_path:
        Path to a JSON file produced by :func:`save_heatmap_json`.

    Returns
    -------
    Dict[int, int]
        Mapping of ``grid_index`` to ``count``.
    """
    if not json_path or not Path(json_path).exists():
        return {}
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {int(d["grid_index"]): int(d["count"]) for d in data}
    except Exception:
        return {}


def build_heatmap(midi_path: str, resolution: int = RESOLUTION) -> Dict[int, int]:
    """Parse a MIDI file and count note onsets per measure.

    Parameters
    ----------
    midi_path:
        Path to the MIDI file to analyse.
    resolution:
        Number of bins per measure. Default is ``RESOLUTION``.

    Returns
    -------
    List[int]
        Flattened list of onset counts. ``resolution`` elements are produced for
        each measure in the MIDI file.
    """
    score = converter.parse(midi_path)

    ts = score.recurse().getElementsByClass(meter.TimeSignature)
    if len(ts) == 0:
        raise RuntimeError("拍子情報が見つかりませんでした。")
    first_ts = ts[0]
    beats_per_measure = first_ts.numerator
    beat_unit = first_ts.denominator
    quarter_per_measure = (beats_per_measure * (4.0 / beat_unit)) / 2

    onset_offsets: List[float] = []
    for el in score.recurse():
        if isinstance(el, note.Note) or isinstance(el, chord.Chord):
            onset_offsets.append(float(el.offset))

    if len(onset_offsets) == 0:
        raise RuntimeError("ノート（Note/Chord）のオンセットが見つかりませんでした。")

    max_offset = max(onset_offsets)
    total_measures = int(max_offset // quarter_per_measure) + 1
    heatmap_counts = [0] * (total_measures * resolution)

    for off in onset_offsets:
        measure_index = int(off // quarter_per_measure)
        offset_in_measure = off - (measure_index * quarter_per_measure)
        subbeat_index = int((offset_in_measure / quarter_per_measure) * resolution)
        if subbeat_index >= resolution:
            subbeat_index = resolution - 1
        grid_index = measure_index * resolution + subbeat_index
        if 0 <= grid_index < len(heatmap_counts):
            heatmap_counts[grid_index] += 1

    collapsed: Dict[int, int] = {}
    for idx, cnt in enumerate(heatmap_counts):
        if cnt > 0:
            collapsed[idx % resolution] = collapsed.get(idx % resolution, 0) + cnt
    return collapsed


def save_heatmap_json(midi_path: str, out_json: str, resolution: int = RESOLUTION) -> None:
    """Save heatmap as a JSON file compatible with :func:`load_heatmap_data`."""
    counts = build_heatmap(midi_path, resolution)
    data = [{"grid_index": idx, "count": cnt} for idx, cnt in counts.items()]
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate onset heatmap JSON from a MIDI file")
    parser.add_argument("midi_path", help="Input MIDI file")
    parser.add_argument("resolution", type=int, nargs="?", default=RESOLUTION, help="Bins per measure")
    parser.add_argument("output", nargs="?", default="heatmap.json", help="Output JSON path")
    args = parser.parse_args()

    try:
        save_heatmap_json(args.midi_path, args.output, args.resolution)
    except Exception as e:  # pragma: no cover - CLI feedback
        parser.exit(1, f"Error: {e}\n")
