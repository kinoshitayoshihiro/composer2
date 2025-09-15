# ujam/io.py
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Optional
import re
import logging

from .consts import PITCH_CLASS
from .phrase_schedule import ChordSpan  # type: ignore

# 先頭付近（import 群のあと）
# star import などでテストに流れ込む名前を制御
__all__ = [
    # 公開したい関数・クラスだけを明示（例）
    "ChordSpan",
    "build_sparkle_midi",
    "parse_note_token",
    "parse_midi_note",
    "apply_section_preset",
    "schedule_phrase_keys",
    "vocal_onsets_from_midi",
    "vocal_features_from_midi",
    "main",
]
# ※ 下線始まりの名前は通常 * で公開されませんが、明示しておくと安全

try:
    import yaml  # type: ignore
except Exception:
    yaml = None


def read_chords_yaml(path: Path) -> List["ChordSpan"]:
    """
    YAML -> List[ChordSpan]
    - 必須キー欠損: KeyError
    - 不明なルート名: ValueError
    """
    if yaml is None:
        raise SystemExit("PyYAML is required to read YAML chord files. pip install pyyaml")

    text = path.read_text()
    # Allow compact mappings like "start:0" without a space after ':'
    text = re.sub(r":(?=\S)", ": ", text)
    raw = yaml.safe_load(text)
    if raw is None:
        return []

    if isinstance(raw, dict):
        items = [raw]
    elif isinstance(raw, list):
        items = raw
    else:
        raise ValueError("Chord YAML must be a list or mapping")

    spans: List[ChordSpan] = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"Chord YAML item {i} must be a mapping")
        missing = [k for k in ("start", "end", "root") if k not in item]
        if missing:
            # テスト期待: 欠損は KeyError
            raise KeyError(missing[0])
        try:
            start = float(item["start"])
            end = float(item["end"])
        except Exception as e:
            raise ValueError("start/end must be numeric") from e

        root = str(item["root"]).strip()
        quality = str(item.get("quality", "maj")).strip().lower()
        if root not in PITCH_CLASS:
            # テスト期待: 不明ルートは ValueError
            raise ValueError(f"Unknown root {root}")

        spans.append(ChordSpan(start, end, PITCH_CLASS[root], quality))
    return spans


NOTE_RE = re.compile(r"^([A-G][b#]?)(-?\d+)$")


def _parse_note(
    tok, aliases: Optional[Dict[str, int]] = None, *, warn_unknown: bool = False
) -> Optional[int]:
    """
    内部ユーティリティ:
    - "rest" -> None
    - エイリアス優先
    - C#1 等の名表記 or 整数(0..127)
    - warn_unknown=True かつ識別子ぽい未登録名は warning を出して None を返す
    """
    if tok is None:
        return None
    if isinstance(tok, int):
        if 0 <= tok <= 127:
            return tok
        raise ValueError("note out of range")

    if isinstance(tok, str):
        t = tok.strip()
        if t.lower() in {"rest", "silence", "none", "-"}:
            return None
        if aliases and t in aliases:
            return aliases[t]
        if aliases and warn_unknown and t.isidentifier() and t not in aliases:
            logging.warning("unknown note alias: %s", t)
            return None

        m = NOTE_RE.match(t)
        if m:
            name, octv = m.groups()
            pc = PITCH_CLASS.get(name)
            if pc is None:
                raise ValueError(f"Unknown pitch class {name}")
            val = (int(octv) + 1) * 12 + pc
            if not (0 <= val <= 127):
                raise ValueError("note out of range")
            return val

        try:
            v = int(t)
        except Exception as e:
            raise ValueError("invalid note token") from e
        if not (0 <= v <= 127):
            raise ValueError("note out of range")
        return v

    raise ValueError("invalid note token")


def read_section_profiles(path: Path, aliases: Optional[Dict[str, int]] = None) -> List[Dict]:
    """
    セクション設定の読込:
    {
      sections: [
        { tag, bars:[start,end], phrase_pool:[...], pool_weights:[], fill_cadence:int,
          density_curve:'flat'|'rise'|'fall'|'s-curve', accent_arc:[lo,hi], density:'low'|'med'|'high' }
      ]
    }
    - phrase_pool は _parse_note で解釈（rest/alias/名表記/整数）
    """
    text = path.read_text()
    if yaml is None:
        import json

        data = json.loads(text)
    else:
        data = yaml.safe_load(text) or {}

    sections = data.get("sections")
    if not isinstance(sections, list):
        raise ValueError("sections must be a list")

    out: List[Dict] = []
    for idx, sec in enumerate(sections):
        if not isinstance(sec, dict):
            raise ValueError(f"sections[{idx}] must be mapping")

        tag = sec.get("tag")
        if not isinstance(tag, str):
            raise ValueError(f"sections[{idx}].tag required")

        bars = sec.get("bars")
        if not (isinstance(bars, list) and len(bars) == 2):
            raise ValueError(f"sections[{idx}].bars must be [start,end]")
        start_bar, end_bar = int(bars[0]), int(bars[1])

        pool = sec.get("phrase_pool")
        if not (isinstance(pool, list) and pool):
            raise ValueError(f"sections[{idx}].phrase_pool required")
        try:
            pool_notes = [_parse_note(tok, aliases, warn_unknown=True) for tok in pool]
        except Exception as e:
            raise ValueError(f"sections[{idx}].phrase_pool: {e}") from e

        weights = sec.get("pool_weights")
        if weights is not None:
            if not (isinstance(weights, list) and len(weights) == len(pool_notes)):
                raise ValueError(f"sections[{idx}].pool_weights length mismatch")
            weights = [float(w) for w in weights]

        fill_cadence = sec.get("fill_cadence")
        if fill_cadence is not None:
            fc = int(fill_cadence)
            if fc <= 0:
                raise ValueError(f"sections[{idx}].fill_cadence must be >0")
            fill_cadence = fc

        density_curve = sec.get("density_curve")
        if density_curve is not None and density_curve not in ("flat", "rise", "fall", "s-curve"):
            raise ValueError(f"sections[{idx}].density_curve invalid")

        accent_arc = sec.get("accent_arc")
        if accent_arc is not None:
            if not (isinstance(accent_arc, list) and len(accent_arc) == 2):
                raise ValueError(f"sections[{idx}].accent_arc must be [lo,hi]")
            accent_arc = [float(accent_arc[0]), float(accent_arc[1])]

        density = sec.get("density")
        if density is not None and density not in ("low", "med", "high"):
            raise ValueError(f"sections[{idx}].density invalid")

        out.append(
            {
                "tag": tag,
                "start_bar": start_bar,
                "end_bar": end_bar,
                "phrase_pool": pool_notes,
                "pool_weights": weights,
                "fill_cadence": fill_cadence,
                "density_curve": density_curve,
                "accent_arc": accent_arc,
                "density": density,
            }
        )
    return out
