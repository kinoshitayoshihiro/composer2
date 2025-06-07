import music21
from music21 import stream, tempo, meter, key, instrument as m21instrument
import sys
import os
import argparse
import logging
from pathlib import Path

# modular_composer.py の冒頭付近に追加
import utilities.humanizer as humanizer

# --- config_loaderのインポート ---
try:
    from utilities.config_loader import load_main_cfg
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import config_loader: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# --- 各ジェネレーターのインポート ---
from generator.piano_generator import PianoGenerator
from generator.guitar_generator import GuitarGenerator
from generator.bass_generator import BassGenerator
from generator.drum_generator import DrumGenerator
from generator.strings_generator import StringsGenerator
from generator.melody_generator import MelodyGenerator
from generator.sax_generator import SaxGenerator

# --- 役割ディスパッチマップ ---
ROLE_DISPATCH = {
    "melody": MelodyGenerator,
    "counter": MelodyGenerator,
    "pad": StringsGenerator,
    "riff": MelodyGenerator,
    "rhythm": GuitarGenerator,
    "bass": BassGenerator,
    "unison": StringsGenerator,
}


def build_tempo_events(tempo_map_path):
    # tempo_map.json を読み込んで music21 テンポイベントリストを返す
    import json

    with open(tempo_map_path, encoding="utf-8") as f:
        tempo_map = json.load(f)
    events = []
    for entry in tempo_map:
        offset = entry.get("offset_q", 0.0)
        bpm = entry.get("bpm", 120)
        events.append((offset, bpm))
    return events


def main_cli():
    parser = argparse.ArgumentParser(description="OtoKotoba Modular Composer")
    parser.add_argument("--main-cfg", type=Path, default=Path("data/main_cfg.yml"))
    parser.add_argument(
        "--dry-run", action="store_true", help="MIDIを書き出さずロジックのみ実行"
    )
    args = parser.parse_args()

    # 設定ファイルのロード
    main_cfg = load_main_cfg(args.main_cfg)

    # --- 各パートのジェネレーターを役割で自動ディスパッチ ---
    part_gens = {}
    role_dispatch = main_cfg["role_dispatch_factory"]()  # ★ 動的取得
    for part_name, part_cfg in main_cfg["part_defaults"].items():
        role = part_cfg.get("role", "melody")
        GenFactory = role_dispatch[role]
        part_gens[part_name] = GenFactory(
            global_settings=main_cfg.get("global_settings", {}),
            **part_cfg,
            global_time_signature=main_cfg["global_settings"].get(
                "time_signature", "4/4"
            ),
        )
    # --- テンポマップ・拍子チェンジの適用 ---
    score = stream.Score()  # これで score を初期化

    # テンポマップを適用
    if "tempo_map_path" in main_cfg["global_settings"]:
        tempo_events = build_tempo_events(main_cfg["global_settings"]["tempo_map_path"])
        for offset, bpm in tempo_events:
            score.insert(offset, tempo.MetronomeMark(number=bpm))
    else:
        # セクションごとのテンポ/拍子チェンジ
        for sec_name, sec_override in main_cfg.get("section_overrides", {}).items():
            for part_name, gen in part_gens.items():
                part_cfg = main_cfg["part_defaults"].get(part_name, {}).copy()
                # セクションオーバーライドのパート単位上書き
                if part_name in sec_override:
                    part_cfg.update(sec_override[part_name])
                # ここでは例として全体1セクションのみ
                section_data = {
                    "musical_intent": {},
                    "part_params": {},
                    "q_length": 16.0,
                }
                part_stream = gen.compose(section_data=section_data)
                score.append(part_stream)

    # --- 出力 ---
    out_path = Path("output.mid")  # 出力ファイルパスを定義
    if not args.dry_run:
        score.write("midi", fp=out_path)
        print(f"Exported MIDI: {out_path}")
    else:
        print("Dry run: MIDIファイルは書き出しませんでした。")
    if main_cfg["global_settings"].get("humanize_profile"):
        humanizer.apply(score, main_cfg["global_settings"]["humanize_profile"])

    # 例: modular_composer.py メインループ内
    # (3) 全体後処理
    hp3 = main_cfg["global_settings"].get("humanize_profile")
    if hp3:
        humanizer.apply(score, hp3)


if __name__ == "__main__":
    main_cli()
