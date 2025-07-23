"""utilities/prettymidi_sync.py
=================================================
Extract / apply vocal‑groove micro‑timing profiles
-------------------------------------------------
CLI Usage (examples)
--------------------
1) **Extract groove profile** (8‑th grid)
   ```bash
   python utilities/prettymidi_sync.py \
          --mode extract \
          --input data/vocal.mid \
          --subdiv 8 \
          --outfile data/groove.json
   ```

2) **Apply existing profile**
   ```bash
   python utilities/prettymidi_sync.py \
          --mode apply \
          --input drums.mid \
          --groove data/groove.json \
          --outfile drums_synced.mid
   ```

3) **One‑shot** (extract from vocal & apply to band track)
   ```bash
   python utilities/prettymidi_sync.py \
          --mode extract_apply \
          --input band.mid \
          --vocal vocal.mid \
          --subdiv 16 \
          --outfile band_synced.mid
   ```
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import statistics
import sys
import logging
from typing import Dict, List, Tuple
from utilities.onset_heatmap import RESOLUTION

try:
    import pretty_midi
except ImportError as e:
    raise ImportError(
        "pretty_midi module is required. Install it with `pip install pretty_midi`."
    ) from e

# ★★★ music21 の stream をインポート ★★★
from music21 import (
    stream,
    note,
    tempo,
)  # note と tempo も apply_groove_pretty で使う可能性


################################################################################
# ── Grid helpers ──────────────────────────────────────────────────────────────
################################################################################
# (変更なし)
def _sec_per_subdivision(pm: "pretty_midi.PrettyMIDI", subdiv: int) -> float:
    _times, tempi = pm.get_tempo_changes()
    bpm = tempi[0] if len(tempi) else 120.0
    sec_per_beat = 60.0 / bpm
    sec_per_sub = sec_per_beat / (subdiv / 4)
    return sec_per_sub


def _grid_index_and_shift(time_s: float, sec_per_sub: float) -> Tuple[int, float]:
    idx = round(time_s / sec_per_sub)
    return idx, time_s - idx * sec_per_sub


def _collect_note_onsets(pm: "pretty_midi.PrettyMIDI") -> List[float]:
    onsets: List[float] = []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        onsets.extend(n.start for n in inst.notes)
    return sorted(onsets)


################################################################################
# ── Profile IO ────────────────────────────────────────────────────────────────
################################################################################
# (変更なし)
def _write_profile(path: pathlib.Path, data: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _read_profile(path: pathlib.Path) -> Dict:  # pathlib.Path を型ヒントに使用
    return json.loads(path.read_text(encoding="utf-8"))


################################################################################
# ── Extraction ────────────────────────────────────────────────────────────────
################################################################################
# (変更なし)
def extract_groove(pm: "pretty_midi.PrettyMIDI", subdiv: int) -> Dict:
    sec_per_sub = _sec_per_subdivision(pm, subdiv)
    onsets = _collect_note_onsets(pm)
    shifts: List[float] = []
    hist: Dict[str, int] = {}
    for t in onsets:
        _, shift = _grid_index_and_shift(t, sec_per_sub)
        shifts.append(shift)
        bucket = f"{shift:.4f}"
        hist[bucket] = hist.get(bucket, 0) + 1
    _times, tempi = pm.get_tempo_changes()
    bpm = tempi[0] if len(tempi) > 0 else 120.0
    return {
        "bpm": bpm,
        "subdiv": subdiv,
        "sec_per_sub": sec_per_sub,
        "mean_shift_sec": statistics.mean(shifts) if shifts else 0.0,
        "stdev_shift_sec": statistics.stdev(shifts) if len(shifts) > 1 else 0.0,
        "histogram": hist,
    }


################################################################################
# ── Application ───────────────────────────────────────────────────────────────
################################################################################
# (既存の apply_groove は変更なし)
def apply_groove(
    pm: "pretty_midi.PrettyMIDI",
    profile: Dict,
    *,
    strength: float = 1.0,
    min_shift_sec: float = 1e-3,
):
    subdiv = int(profile.get("subdiv", 16))
    sec_per_sub = float(profile.get("sec_per_sub", _sec_per_subdivision(pm, subdiv)))
    hist = profile.get("histogram", {})
    if not hist:
        return
    buckets = [float(k) for k in hist.keys()]
    weights = [hist[k] for k in hist.keys()]
    for inst in pm.instruments:
        for n_note in inst.notes:  # 変数名を n_note に変更 (music21.note との衝突回避)
            idx, _ = _grid_index_and_shift(n_note.start, sec_per_sub)
            target_shift = random.choices(buckets, weights)[0] * strength
            new_start = idx * sec_per_sub + target_shift
            delta = new_start - n_note.start
            if abs(delta) < min_shift_sec:
                continue
            n_note.start += delta
            n_note.end += delta


# --- ★★★ ここからHaruさんのパッチ案を適用 ★★★ ---
def load_groove_profile(
    path_str: str,
) -> Optional[Dict]:  # 引数の型を str に変更 (Pathオブジェクトは呼び出し元で)
    """JSON または YAML のグルーヴプロファイルを読み込むラッパー。"""
    # pathlib.Path を使用するように修正
    path_obj = pathlib.Path(path_str)
    if not path_obj.exists():
        logging.error(f"Groove profile not found at: {path_obj}")
        return None
    try:
        return _read_profile(path_obj)  # 既存の内部関数を再利用
    except Exception as e:
        logging.error(f"Error reading groove profile from {path_obj}: {e}")
        return None


def apply_groove_pretty(
    part: Union[stream.Part, stream.Voice], profile: Dict
) -> Union[stream.Part, stream.Voice]:
    """Apply micro-timing groove to a music21 part."""
    if not profile:
        logging.warning(
            "apply_groove_pretty: Groove profile is empty or None. No groove applied."
        )
        return part

    # --- Deterministic offset map support ---------------------------------
    if all(str(k).isdigit() for k in profile.keys()):
        offset_map = {int(k): float(v) for k, v in profile.items()}
        ql_per_sub = 1.0 / RESOLUTION
        for n_note_m21 in part.recurse().notes:
            idx = int(round(n_note_m21.offset / ql_per_sub))
            n_note_m21.offset += offset_map.get(idx, 0.0)
        logging.info(
            f"Applied deterministic groove map to part '{part.id if part.id else ''}'."
        )
        return part

    # mean_shift_ms と stdev_ms はHaruさんの提案。profile.json に合わせる。
    # groove_profile.json のキー名に合わせる (mean_shift_sec, stdev_shift_sec)
    mean_shift_sec = profile.get("mean_shift_sec", 0.0)
    stdev_shift_sec = profile.get("stdev_shift_sec", 0.05)  # 50ms = 0.05s

    # テンポの取得 (music21.stream.Part から)
    # パートの最初のテンポマーカーを使用する。なければ120BPMを仮定。
    current_tempo_bpm = 120.0
    tempo_marks = part.getElementsByClass(tempo.MetronomeMark)
    if tempo_marks:
        current_tempo_bpm = tempo_marks[0].number

    sec_per_ql = 60.0 / current_tempo_bpm  # 1拍（4分音符）あたりの秒数

    if sec_per_ql <= 0:  # 無効なテンポの場合
        logging.warning(
            "apply_groove_pretty: Invalid tempo (sec_per_ql <= 0). Cannot apply groove."
        )
        return part

    for n_note_m21 in part.flatten().notes:  # 変数名を n_note_m21 に変更
        # 既に適用されたオフセットを考慮せず、元のグリッドからのズレとして計算する方が良い場合もあるが、
        # ここでは単純に現在のオフセットに揺らぎを加える。
        jitter_sec = random.gauss(mean_shift_sec, stdev_shift_sec)
        jitter_ql = jitter_sec / sec_per_ql

        original_offset = n_note_m21.offset
        new_offset = original_offset + jitter_ql

        # オフセットが負にならないように調整（オプション）
        if new_offset < 0:
            new_offset = 0.0  # または original_offset を維持

        n_note_m21.offset = new_offset
        # デュレーションは変更しない（タイミングのみの揺らぎ）
        # もしデュレーションも揺らがせたい場合は別途ロジック追加
    logging.info(
        f"Applied simple groove (timing jitter) to part '{part.id if part.id else ''}'. Mean: {mean_shift_sec*1000:.1f}ms, Stdev: {stdev_shift_sec*1000:.1f}ms, Tempo: {current_tempo_bpm} BPM"
    )
    return part


# --- ★★★ パッチ適用ここまで ★★★ ---


################################################################################
# ── CLI ───────────────────────────────────────────────────────────────────────
################################################################################
# (main関数は変更なし)
def main():
    p = argparse.ArgumentParser(
        description="Vocal groove extractor / applier (PrettyMIDI)"
    )
    p.add_argument(
        "--mode", choices=["extract", "apply", "extract_apply"], required=True
    )
    p.add_argument("--input", required=True, help="Input MIDI file (for extract/apply)")
    p.add_argument("--vocal", help="Vocal MIDI (when extract_apply)")
    p.add_argument("--groove", help="Groove JSON (when apply)")
    p.add_argument("--subdiv", type=int, default=16)
    p.add_argument("--outfile", required=True)
    p.add_argument("--strength", type=float, default=1.0, help="Groove strength 0‑1")
    p.add_argument(
        "--quantize", type=float, default=1e-3, help="Minimum shift to apply (sec)"
    )
    args = p.parse_args()
    in_path = pathlib.Path(args.input)
    out_path = pathlib.Path(args.outfile)
    if args.mode == "extract":
        pm = pretty_midi.PrettyMIDI(str(in_path))
        prof = extract_groove(pm, args.subdiv)
        _write_profile(out_path, prof)
        print(f"[Groove] profile extracted to {out_path}")
    elif args.mode == "apply":
        if not args.groove:
            p.error("--groove required in apply mode")
        prof = load_groove_profile(
            args.groove
        )  # ★ _read_profile -> load_groove_profile に変更
        if not prof:
            sys.exit(1)  # プロファイルがロードできなかった場合
        pm = pretty_midi.PrettyMIDI(str(in_path))
        apply_groove(pm, prof, strength=args.strength, min_shift_sec=args.quantize)
        pm.write(str(out_path))
        print(f"[Groove] applied groove -> {out_path}")
    elif args.mode == "extract_apply":
        if not args.vocal:
            p.error("--vocal required in extract_apply mode")
        vocal_pm = pretty_midi.PrettyMIDI(str(pathlib.Path(args.vocal)))
        prof = extract_groove(vocal_pm, args.subdiv)
        band_pm = pretty_midi.PrettyMIDI(str(in_path))
        apply_groove(band_pm, prof, strength=args.strength, min_shift_sec=args.quantize)
        band_pm.write(str(out_path))
        print(f"[Groove] extracted from {args.vocal} and applied -> {out_path}")


if __name__ == "__main__":
    main()
# --- END OF FILE utilities/prettymidi_sync.py ---
