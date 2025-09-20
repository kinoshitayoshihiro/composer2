#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
chordmap_merge.py — Merge a Base Chordmap (既存曲の背骨) and a Narrative Chordmap（歌詞/感情の彩り）
into a final chordmap with explicit, conservative rules.

依存: PyYAML のみ（music21等は使いません）
使い方（単一YAMLに全部入っている場合）:
    python -m utilities.chordmap_merge all_in_one.yaml --out chordmap.final.yaml

分割ファイルの場合:
    python -m utilities.chordmap_merge \
      --base base_chordmap.yaml \
      --narr narrative_chords.yaml \
      --policy merge_policy.yaml \
      --context context.yaml \
      --out chordmap.final.yaml

入力YAML（いずれかの形）:
  [A] 単一ファイル:
    key: D major
    meter: 4/4
    tempo_map: tempo_map.json
    sections: [...]
    base_chords: [{bar: 1, root: D, quality: maj7, dur: 1, bass: null, tensions: [9]} ...]
    narrative_chords:
      - {bar: 2, intent: warm,  op: add_tension, add: ["9"], weight: 1.0}
      - {bar: 4, intent: lift,  op: slash, bass: F#}
      - {bar: 6, intent: wistful, op: substitute, to: {root: G, quality: maj7}, weight: 0.5}
      - {bar: 8, intent: resolve, op: cadence_lock}
    merge_policy:
      root_change_allowed: false
      root_change_bars: []           # 例外的にRoot変更を許す小節番号
      allow_substitutions: ["I↔vi","IV↔ii","V→V7sus4","ii→ii7"]  # 参考（本実装は品質テキストのみ参照）
      strong_beat_lock: true
      bass_guard: true
      tension_limits: {max_stack: 2}
      weight_threshold: 0.5

  [B] 分割ファイル:
    --base      : { key?, meter?, tempo_map?, sections?, base_chords: [...] }
    --narr      : { narrative_chords: [...] }
    --policy    : { merge_policy: {...} }
    --context   : { key?, meter?, tempo_map?, sections? }  # 任意（不足メタの補完）

出力:
  key, meter, tempo_map, sections を引き継ぎ、 final_chords を生成して書き出す。
  final_chords: [{bar, root, quality, dur, bass?, tensions?}, ...]

マージ規則（要点）:
- 小節境界と根音は Base 優先。
- Narrative は原則「色付け」(tensions / slash / quality) を優先。
- 置換(substitute)での root 変更は、policy.root_change_allowed または root_change_bars にある場合のみ許可。
- strong_beat_lock が true の場合、bar 頭の root 変更は抑止（彩りのみ適用）。
- bass_guard が true の場合、既存 bass と矛盾する slash は無効化。
- tension_limits.max_stack を超えるテンション追加はクリップ。
- narrativeの weight（0..1）が weight_threshold 未満の場合はスキップ。
- cadence_lock が入った bar は、以降のその小節に対する root/quality 変更を抑止（色付けのみ可）。

最小実装のため、機能はシンプルで安全側に倒しています（不整合は Base にフォールバック）。
"""

from __future__ import annotations
import argparse
from typing import Any, Dict, List, Optional, Tuple
import copy
import yaml

DEFAULT_POLICY = {
    "root_change_allowed": False,
    "root_change_bars": [],
    "allow_substitutions": ["I↔vi", "IV↔ii", "V→V7sus4", "ii→ii7"],
    "strong_beat_lock": True,
    "bass_guard": True,
    "tension_limits": {"max_stack": 2},
    "weight_threshold": 0.5,
}

ChordRec = Dict[str, Any]


def _coerce_base_chords(base_list: List[Dict[str, Any]]) -> List[ChordRec]:
    """Baseの各要素に最低限のキーを用意。未知キーは温存。"""
    out: List[ChordRec] = []
    for it in base_list:
        rec = {
            "bar": int(it.get("bar")),
            "root": str(it.get("root")),
            "quality": str(it.get("quality", "")),
            "dur": it.get("dur", 1),
        }
        # 任意フィールド
        if "bass" in it and it["bass"] is not None:
            rec["bass"] = str(it["bass"])
        tensions = it.get("tensions")
        if tensions:
            rec["tensions"] = list(dict.fromkeys([str(x) for x in tensions]))
        out.append(rec)
    # bar昇順
    out.sort(key=lambda r: (r["bar"]))
    return out


def _group_by_bar(chords: List[ChordRec]) -> Dict[int, ChordRec]:
    """bar → 代表和音（本実装は bar 単位で1和音想定）"""
    return {c["bar"]: copy.deepcopy(c) for c in chords}


def _apply_add_tension(rec: ChordRec, add: List[str], policy: Dict[str, Any]) -> None:
    if not add:
        return
    max_stack = int(policy.get("tension_limits", {}).get("max_stack", 2))
    cur = list(rec.get("tensions", []))
    for t in add:
        ts = str(t)
        if ts not in cur:
            cur.append(ts)
    # クリップ
    if len(cur) > max_stack:
        cur = cur[:max_stack]
    if cur:
        rec["tensions"] = cur
    else:
        rec.pop("tensions", None)


def _apply_slash(rec: ChordRec, bass: Optional[str], policy: Dict[str, Any]) -> None:
    if not bass:
        return
    if policy.get("bass_guard", True) and "bass" in rec and rec["bass"] is not None:
        # 既存と矛盾する場合は無視（安全側）
        if str(rec["bass"]) != str(bass):
            return
    rec["bass"] = str(bass)


def _apply_substitute(
    rec: ChordRec, to: Dict[str, Any], policy: Dict[str, Any], allow_root_change: bool
) -> None:
    """quality / root を置換。root変更は許可された場合のみ。"""
    if not to:
        return
    to_root = str(to.get("root", rec["root"])) if to.get("root") else rec["root"]
    to_quality = str(to.get("quality", rec.get("quality", "")))
    # root 変更可否
    if (to_root != rec["root"]) and (not allow_root_change):
        # rootは据え置き、qualityのみ変更
        rec["quality"] = to_quality
        return
    # 許容
    rec["root"] = to_root
    rec["quality"] = to_quality
    # rootが変わるなら slash/bass は無効化（矛盾を避ける）
    if to_root != rec.get("root"):
        rec.pop("bass", None)


def _should_skip_by_weight(n: Dict[str, Any], policy: Dict[str, Any]) -> bool:
    thr = float(policy.get("weight_threshold", 0.5))
    w = float(n.get("weight", 1.0))
    return w < thr


def _merge_one(
    bar: int,
    base: ChordRec,
    edits: List[Dict[str, Any]],
    policy: Dict[str, Any],
    strong_beat: bool,
    locked: bool,
) -> ChordRec:
    """単一 bar に対して Narrative 編集を順次適用。強拍/ロック/ルールに従い安全に処理。"""
    out = copy.deepcopy(base)
    for n in edits:
        if _should_skip_by_weight(n, policy):
            continue
        op = str(n.get("op", "")).strip()
        if op == "cadence_lock":
            # ロック命令自体はここでは処理済み（呼び出し側で bar を収集）
            continue

        # root 変更許容判定
        root_change_allowed = bool(policy.get("root_change_allowed", False))
        if bar in policy.get("root_change_bars", []):
            root_change_allowed = True
        # 強拍ロック：bar頭では root変更禁止（彩りのみOK）
        if policy.get("strong_beat_lock", True) and strong_beat:
            root_change_allowed = False
        # 小節がすでに lock 済みなら root/quality変更を禁止
        if locked:
            # 彩り（tension/slash）のみ許可
            if op == "add_tension":
                _apply_add_tension(out, n.get("add", []), policy)
            elif op == "slash":
                _apply_slash(out, n.get("bass"), policy)
            # substitute はスキップ
            continue

        if op == "add_tension":
            _apply_add_tension(out, n.get("add", []), policy)
        elif op == "slash":
            _apply_slash(out, n.get("bass"), policy)
        elif op == "substitute":
            _apply_substitute(out, n.get("to", {}), policy, allow_root_change=root_change_allowed)
        else:
            # 未知の op は無視（安全側）
            continue
    return out


def merge_chordmaps(
    base_chords: List[ChordRec], narrative: List[Dict[str, Any]], policy: Dict[str, Any]
) -> List[ChordRec]:
    """Base × Narrative → Final（bar単位の最小実装）"""
    base = _coerce_base_chords(base_chords)
    base_by_bar = _group_by_bar(base)
    # bar → narrative edits
    narr_by_bar: Dict[int, List[Dict[str, Any]]] = {}
    locked_bars: set[int] = set()
    for n in narrative or []:
        b = int(n.get("bar", -1))
        if b < 0:
            continue
        narr_by_bar.setdefault(b, []).append(n)
        if str(n.get("op", "")) == "cadence_lock":
            locked_bars.add(b)

    # 各barに適用
    final_list: List[ChordRec] = []
    for b in sorted(base_by_bar.keys()):
        edits = narr_by_bar.get(b, [])
        # 本実装は bar 頭のみ扱うため strong_beat=True を適用（beat内編集が必要なら拡張）
        strong_beat = True
        locked = b in locked_bars
        merged = _merge_one(b, base_by_bar[b], edits, policy, strong_beat, locked)
        # 安全のため最低限の整形
        merged["bar"] = b
        if "dur" not in merged:
            merged["dur"] = base_by_bar[b].get("dur", 1)
        final_list.append(merged)
    return final_list


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _dump_yaml(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def _gather_inputs(
    args,
) -> Tuple[Dict[str, Any], List[ChordRec], List[Dict[str, Any]], Dict[str, Any]]:
    """
    戻り値: (context_meta, base_chords, narrative_chords, policy)
    context_meta: key/meter/tempo_map/sections を含む可能性のある dict
    """
    # 単一ファイルに全部あるケース
    if args.all_in_one:
        d = _load_yaml(args.all_in_one)
        meta = {k: d.get(k) for k in ("key", "meter", "tempo_map", "sections") if k in d}
        base = d.get("base_chords", [])
        narr = d.get("narrative_chords", [])
        pol = d.get("merge_policy", {})
        return meta, base, narr, pol

    # 分割ファイル
    meta: Dict[str, Any] = {}
    base = []
    narr = []
    pol = {}

    if args.base:
        db = _load_yaml(args.base)
        base = db.get("base_chords", db.get("chords", []))
        for k in ("key", "meter", "tempo_map", "sections"):
            if k in db and k not in meta:
                meta[k] = db[k]
    if args.narr:
        dn = _load_yaml(args.narr)
        narr = dn.get("narrative_chords", dn.get("edits", []))
    if args.policy:
        dp = _load_yaml(args.policy)
        pol = dp.get("merge_policy", dp)
    if args.context:
        cx = _load_yaml(args.context)
        for k in ("key", "meter", "tempo_map", "sections"):
            if k in cx and k not in meta:
                meta[k] = cx[k]
    return meta, base, narr, pol


def main():
    ap = argparse.ArgumentParser(
        description="Merge Base and Narrative chordmaps into final_chords."
    )
    ap.add_argument(
        "all_in_one",
        nargs="?",
        default=None,
        help="単一YAML（meta+base_chords+narrative_chords+merge_policy を含む）",
    )
    ap.add_argument("--base", type=str, default=None, help="Base chordmap YAML")
    ap.add_argument("--narr", type=str, default=None, help="Narrative edits YAML")
    ap.add_argument("--policy", type=str, default=None, help="Merge policy YAML")
    ap.add_argument(
        "--context", type=str, default=None, help="Meta context YAML (key/meter/tempo_map/sections)"
    )
    ap.add_argument("--out", type=str, default="chordmap.final.yaml")
    args = ap.parse_args()

    meta, base, narr, pol = _gather_inputs(args)

    # デフォルトポリシーで上書き
    policy = copy.deepcopy(DEFAULT_POLICY)
    policy.update(pol or {})

    if not base:
        raise SystemExit(
            "ERROR: base_chords が見つかりません。--base か 単一YAMLを指定してください。"
        )

    final_chords = merge_chordmaps(base, narr or [], policy)

    out = {}
    out.update({k: v for k, v in meta.items() if v is not None})
    out["final_chords"] = final_chords

    _dump_yaml(args.out, out)
    print(f"Wrote {args.out} with {len(final_chords)} bars.")


if __name__ == "__main__":
    main()
