# utilities/tempo_loader.py
import json
from pathlib import Path
from typing import List, Tuple


def load_tempo_map(path: Path) -> List[Tuple[float, int]]:
    """
    JSON ファイルを読み込み、(offset_q, bpm) のリストを返す。
    offset_q: 小節ではなく四分音符基準の絶対オフセット
    bpm: テンポ
    """
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    events = []
    for e in data:
        try:
            off = float(e["offset_q"])
            bpm = int(e["bpm"])
            events.append((off, bpm))
        except (KeyError, ValueError):
            continue
    # オフセット昇順でソートしておく
    return sorted(events, key=lambda x: x[0])
