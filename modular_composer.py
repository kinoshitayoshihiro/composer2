# modular_composer.py  (re‑written 2025‑06‑08)
# =========================================================
# - chordmap(YAML) をロード
# - 各楽器ごとに 1 つだけ Part を用意
# - chordmap に含まれる **absolute_offset_beats** を唯一の座標系として採用
#   * Generator には "セクション内 0 拍起点" でデータを渡す
#   * Score へ入れる時のみ section_start_q を加算
# - Humanizer を part → global の順に適用
# ---------------------------------------------------------

from __future__ import annotations

import argparse
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict
from utilities.config_loader import load_chordmap_yaml, load_main_cfg
from music21 import instrument as m21inst, meter, stream, tempo

# --- project utilities ----------------------------------------------------
from utilities.generator_factory import GenFactory  # type: ignore
import utilities.humanizer as humanizer  # type: ignore

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# -------------------------------------------------------------------------
# helper
# -------------------------------------------------------------------------


def clone_element(elem):
    """music21 オブジェクトを安全に複製して返す"""
    try:
        return elem.clone()
    except Exception:
        return deepcopy(elem)


def normalise_chords_to_relative(chords: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    """chord_event list 内の *_offset_beats をセクション内相対 0.0 起点に変換"""
    if not chords:
        return chords
    origin = chords[0]["absolute_offset_beats"]
    rel_chords = deepcopy(chords)
    for evt in rel_chords:
        for key in (
            "original_offset_beats",
            "humanized_offset_beats",
            "absolute_offset_beats",
        ):
            if key in evt:
                evt[key] -= origin
    return rel_chords


# -------------------------------------------------------------------------
# main
# -------------------------------------------------------------------------


def main_cli() -> None:
    parser = argparse.ArgumentParser(description="OtoKotoba Modular Composer")
    parser.add_argument("--main-cfg", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # 1) 設定 & データロード -------------------------------------------------
    main_cfg = load_main_cfg(args.main_cfg)
    chordmap = load_chordmap_yaml(Path(main_cfg["paths"]["chordmap_path"]))

    section_names: list[str] = main_cfg["sections_to_generate"]
    raw_sections: dict[str, Dict[str, Any]] = chordmap["sections"]
    sections = [raw_sections[n] for n in section_names if n in raw_sections]
    if not sections:
        logging.error("指定セクションが chordmap に見つかりませんでした")
        return

    ts = meter.TimeSignature(main_cfg["global_settings"].get("time_signature", "4/4"))
    beats_per_measure = ts.numerator

    # 2) Generator 初期化 ----------------------------------------------------
    part_gens = GenFactory.build_from_config(main_cfg)

    # 楽器ごとの単一 Part
    part_streams: dict[str, stream.Part] = {}
    for part_name in part_gens:
        p = stream.Part(id=part_name)
        try:
            p.insert(0, m21inst.fromString(part_name))
        except Exception:
            p.partName = part_name
        part_streams[part_name] = p

    # 3) セクション毎にフレーズ生成 -----------------------------------------
    for sec in sections:
        label = sec["label"]
        chords_abs: list[Dict[str, Any]] = sec["processed_chord_events"]
        if not chords_abs:
            logging.warning(f"Section '{label}' に chord イベントがありません")
            continue

        # セクション開始拍 (曲全体の絶対値)
        section_start_q = chords_abs[0]["absolute_offset_beats"]

        # セクション内相対オフセットへ正規化
        chords_rel = normalise_chords_to_relative(chords_abs)

        # セクション長を実データから算出
        last_evt = chords_abs[-1]
        length_q = (
            last_evt["absolute_offset_beats"]
            + last_evt["humanized_duration_beats"]
            - section_start_q
        )
        # フォールバック
        if length_q <= 0:
            length_q = sec.get("length_in_measures", 0) * beats_per_measure

        base_sec_data: Dict[str, Any] = {
            "musical_intent": {
                "section_name": label,
                "chords": chords_rel,
                "emotion": sec["musical_intent"].get("emotion", "neutral"),
                "start_q": 0.0,  # Generator 視点では 0 から始まる
                "vocal_notes": [],
            },
            "q_length": length_q,
        }

        for part_name, gen in part_gens.items():
            part_cfg = main_cfg["part_defaults"].get(part_name, {})
            sec_data = deepcopy(base_sec_data)
            sec_data["part_params"] = part_cfg

            part_sec_stream = gen.compose(section_data=sec_data)
            for elem in part_sec_stream.flatten().notesAndRests:
                part_streams[part_name].insert(
                    section_start_q + elem.offset, clone_element(elem)
                )

    # 4) Humanizer -----------------------------------------------------------
    for name, p_stream in part_streams.items():
        prof = main_cfg["part_defaults"].get(name, {}).get("humanize_profile")
        if prof:
            humanizer.apply(p_stream, prof)

    score = stream.Score(list(part_streams.values()))
    global_prof = main_cfg["global_settings"].get("humanize_profile")
    if global_prof:
        humanizer.apply(score, global_prof)

    # 5) Tempo マップ & 書き出し -------------------------------------------
    tempo_map_path = main_cfg["global_settings"].get("tempo_map_path")
    if tempo_map_path:
        from utilities.tempo_loader import load_tempo_map  # type: ignore

        for off_q, bpm in load_tempo_map(Path(tempo_map_path)):
            score.insert(off_q, tempo.MetronomeMark(number=bpm))

    out_dir = Path(main_cfg["paths"].get("output_dir", "midi_output"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_filename = main_cfg["paths"].get("output_filename", "output.mid")
    out_path = out_dir / out_filename

    if args.dry_run:
        logging.info(f"Dry run – MIDI は書き出しません ({out_path})")
    else:
        try:
            score.write("midi", fp=str(out_path))
            print(f"Exported MIDI: {out_path}")
        except Exception as e:
            logging.error(f"MIDI 書き出し失敗: {e}")


if __name__ == "__main__":
    main_cli()
