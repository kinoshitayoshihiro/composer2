from collections import Counter
from typing import Dict, List
import pretty_midi, json

RESOLUTION = 16  # 16分=1グリッド


def build_heatmap(midi_path: str, resolution: int = RESOLUTION) -> Dict[int, int]:
    """MIDI からオンセット数ヒートマップ {grid_index:count}."""
    pm = pretty_midi.PrettyMIDI(midi_path)
    ticks_per_beat = pm.time_to_tick(1)  # 1 秒ではなく1拍分 tick
    grid_ticks = ticks_per_beat / resolution
    counter: Counter = Counter()
    for inst in pm.instruments:
        for n in inst.notes:
            idx = int(n.start / pm.tick_to_time(grid_ticks))
            counter[idx % resolution] += 1
    return dict(counter)


def save_heatmap_json(midi_path: str, out_json: str):
    with open(out_json, "w") as f:
        json.dump(build_heatmap(midi_path), f, indent=2)
