# modular_composer.py  (re‑written 2025‑06‑08)
# =========================================================
# - chordmap(YAML) をロード
# - 各楽器ごとに 1 つだけ Part を用意
# - chordmap に含まれる **absolute_offset_beats** を唯一の座標系として採用
#   * Generator には "セクション内 0 拍点" でデータを渡す
#   * Score へ入れる時のみ section_start_q を加算
# - Humanizer を part → global の順に適用
# ---------------------------------------------------------

from __future__ import annotations

import argparse
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any

from music21 import (
    chord,
    dynamics,
    expressions,
    key,
    meter,
    note,
    stream,
    tempo,
)
from music21 import (
    instrument as m21inst,
)

import utilities.humanizer as humanizer  # type: ignore
from utilities.config_loader import load_chordmap_yaml, load_main_cfg

# --- project utilities ----------------------------------------------------
from utilities.generator_factory import GenFactory  # type: ignore
from utilities.rhythm_library_loader import load_rhythm_library
from utilities.tempo_utils import load_tempo_map
from generator.guitar_generator import TUNING_PRESETS

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def configure_logging(args: argparse.Namespace) -> None:
    """Configure global logging level based on CLI args."""
    level = logging.WARNING
    if getattr(args, "verbose", False):
        level = logging.INFO
    if getattr(args, "log_level", None):
        level = getattr(logging, args.log_level)
    logging.getLogger().setLevel(level)
    logger.setLevel(level)


# -------------------------------------------------------------------------
# helper
# -------------------------------------------------------------------------


def clone_element(elem):
    """music21 オブジェクトを安全に複製して返す"""
    try:
        return elem.clone()
    except Exception:
        return deepcopy(elem)


def normalise_chords_to_relative(chords: list[dict[str, Any]]) -> list[dict[str, Any]]:
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


def compose(
    main_cfg: dict[str, Any],
    chordmap: dict[str, Any],
    rhythm_lib,
    overrides_model: Any | None = None,
    tempo_map=None,
    num_workers: int = 1,
    use_processes: bool = False,
    buffer_ahead: int = 0,
    parallel_bars: int = 1,
) -> tuple[stream.Score, list[dict[str, Any]]]:
    if tempo_map is None:
        part_gens = GenFactory.build_from_config(main_cfg, rhythm_lib)
    else:
        part_gens = GenFactory.build_from_config(
            main_cfg, rhythm_lib, tempo_map=tempo_map
        )

    part_streams: dict[str, stream.Part] = {}
    used_ids: set[str] = set()

    sections_to_gen: list[str] = main_cfg["sections_to_generate"]
    raw_sections: dict[str, dict[str, Any]] = chordmap.get("sections", {})
    sections: list[dict[str, Any]] = []
    for name in sections_to_gen:
        sec = raw_sections.get(name)
        if not sec:
            continue
        sec_copy = dict(sec)
        sec_copy["label"] = name
        sections.append(sec_copy)

    executor = None
    if num_workers and num_workers > 1:
        if use_processes:
            from concurrent.futures import ProcessPoolExecutor

            executor = ProcessPoolExecutor(max_workers=num_workers)
        else:
            from concurrent.futures import ThreadPoolExecutor

            executor = ThreadPoolExecutor(max_workers=num_workers)

    for sec in sections:
        label = sec["label"]
        chords_abs = sec.get("processed_chord_events", [])
        if not chords_abs:
            continue
        section_start_q = chords_abs[0]["absolute_offset_beats"]
        kick_map: dict[str, list[float]] = {}
        for idx, ch_ev in enumerate(chords_abs):
            next_ev = chords_abs[idx + 1] if idx + 1 < len(chords_abs) else None
            block_start = ch_ev["absolute_offset_beats"] - section_start_q
            block_length = ch_ev.get(
                "humanized_duration_beats", ch_ev.get("original_duration_beats", 4.0)
            )

            base_block: dict[str, Any] = {
                "section_name": label,
                "absolute_offset": block_start,
                "q_length": block_length,
                "chord_symbol_for_voicing": ch_ev.get("chord_symbol_for_voicing"),
                "specified_bass_for_voicing": ch_ev.get("specified_bass_for_voicing"),
                "original_chord_label": ch_ev.get("original_chord_label"),
            }

            next_block = None
            if next_ev:
                next_block = {
                    "chord_symbol_for_voicing": next_ev.get("chord_symbol_for_voicing"),
                    "specified_bass_for_voicing": next_ev.get(
                        "specified_bass_for_voicing"
                    ),
                    "original_chord_label": next_ev.get("original_chord_label"),
                    "q_length": next_ev.get(
                        "humanized_duration_beats",
                        next_ev.get("original_duration_beats", 4.0),
                    ),
                }

            results: dict[str, Any] = {}
            for part_name, gen in part_gens.items():
                part_cfg = main_cfg["part_defaults"].get(part_name, {})
                blk = deepcopy(base_block)
                blk["part_params"] = part_cfg
                blk.setdefault("shared_tracks", {})["kick_offsets"] = [
                    o for lst in kick_map.values() for o in lst
                ]
                if executor:
                    results[part_name] = executor.submit(
                        gen.compose,
                        section_data=blk,
                        overrides_root=overrides_model,
                        next_section_data=next_block,
                        shared_tracks=blk["shared_tracks"],
                    )
                else:
                    results[part_name] = gen.compose(
                        section_data=blk,
                        overrides_root=overrides_model,
                        next_section_data=next_block,
                        shared_tracks=blk["shared_tracks"],
                    )

            for part_name, res in results.items():
                gen = part_gens[part_name]
                result = res.result() if executor else res
                if hasattr(gen, "get_kick_offsets"):
                    kick_map[part_name] = gen.get_kick_offsets()

                if isinstance(result, dict):
                    items = list(result.items())
                elif isinstance(result, (list, tuple)):
                    seq = list(result)
                    items = []
                    for i, sub in enumerate(seq):
                        pid = getattr(sub, "id", None)
                        if not pid:
                            pid = f"{part_name}_{i}"
                            try:
                                sub.id = pid
                            except Exception:
                                pass
                        items.append((pid, sub))
                else:
                    pid = getattr(result, "id", None)
                    if not pid:
                        pid = f"{part_name}_0"
                        try:
                            result.id = pid
                        except Exception:
                            pass
                    items = [(pid, result)]

                fixed_items = []
                for base_pid, sub_stream in items:
                    pid = base_pid
                    if pid in used_ids or pid == "":
                        base = pid if pid else f"{part_name}_0"
                        suffix = ""
                        count = 0
                        while True:
                            candidate = base + suffix
                            if candidate not in used_ids:
                                pid = candidate
                                break
                            count += 1
                            suffix = "_dup" if count == 1 else f"_dup{count}"
                    used_ids.add(pid)
                    try:
                        if getattr(sub_stream, "id", None) in (None, ""):
                            sub_stream.id = pid
                    except Exception:
                        pass
                    fixed_items.append((pid, sub_stream))

                for pid, sub_stream in fixed_items:
                    if pid not in part_streams:
                        p = stream.Part(id=pid)
                        try:
                            p.insert(0, m21inst.fromString(pid))
                        except Exception:
                            p.partName = pid
                        part_streams[pid] = p
                    dest = part_streams[pid]
                    has_inst = bool(
                        dest.recurse().getElementsByClass(m21inst.Instrument)
                    )
                    inserted_inst = False
                    for note_elem in sub_stream.recurse():
                        if isinstance(note_elem, m21inst.Instrument):
                            if not has_inst and not inserted_inst:
                                dest.insert(0.0, clone_element(note_elem))
                                inserted_inst = True
                            continue
                        if isinstance(
                            note_elem,
                            (
                                note.GeneralNote,
                                chord.Chord,
                                note.Rest,
                                tempo.MetronomeMark,
                                key.KeySignature,
                                dynamics.Dynamic,
                                expressions.Expression,
                            ),
                        ):
                            dest.insert(
                                section_start_q + block_start + note_elem.offset,
                                clone_element(note_elem),
                            )

        sec.setdefault("shared_tracks", {})["kick_offsets"] = [
            o for lst in kick_map.values() for o in lst
        ]

    for name, p_stream in part_streams.items():
        prof = main_cfg["part_defaults"].get(name, {}).get("humanize_profile")
        if prof:
            humanizer.apply(
                p_stream,
                prof,
                global_settings={
                    "expr_curve": args.expr_curve,
                    "kick_leak_jitter": args.kick_leak_jitter,
                },
            )

    score = stream.Score(list(part_streams.values()))
    global_prof = main_cfg["global_settings"].get("humanize_profile")
    if global_prof:
        humanizer.apply(
            score,
            global_prof,
            global_settings={
                "expr_curve": args.expr_curve,
                "kick_leak_jitter": args.kick_leak_jitter,
            },
        )

    tempo_map_path = main_cfg["global_settings"].get("tempo_map_path")
    if tempo_map_path:
        for off_q, bpm in load_tempo_map(Path(tempo_map_path)):
            score.insert(off_q, tempo.MetronomeMark(number=bpm))

    if executor:
        executor.shutdown()
    return score, sections


# -------------------------------------------------------------------------
# main
# -------------------------------------------------------------------------


def _parse_tuning(value: str) -> list[int] | str:
    """Parse ``--tuning`` argument.

    Accept a preset name from :data:`TUNING_PRESETS` or six comma-separated
    integer offsets. Raises ``argparse.ArgumentTypeError`` for invalid input.
    """
    if "," in value:
        try:
            offsets = [int(x) for x in value.split(",")]
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "comma-separated offsets must be integers"
            ) from exc
        if len(offsets) != 6:
            raise argparse.ArgumentTypeError(
                "tuning offset list must contain 6 values"
            )
        return offsets
    preset = value.lower()
    if preset not in TUNING_PRESETS:
        raise argparse.ArgumentTypeError(
            f"Unknown tuning preset '{value}'. Available: {', '.join(TUNING_PRESETS)}"
        )
    return preset


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="OtoKotoba Modular Composer")
    # 必須
    p.add_argument(
        "--main-cfg",
        "-m",
        required=True,
        help="YAML: 共通設定ファイル (config/main_cfg.yml)",
    )
    # 任意で上書き可能な入力ファイル
    p.add_argument(
        "--chordmap",
        "-c",
        help="YAML: processed_chordmap_with_emotion.yaml のパス",
    )
    p.add_argument(
        "--rhythm",
        "-r",
        help="YAML: rhythm_library.yml のパス",
    )
    p.add_argument("--time-signature", help="Global time signature e.g. 7/8")
    p.add_argument("--phrase-spec", help="YAML: phrase intensity spec")
    p.add_argument(
        "--insert-phrase",
        action="append",
        help="Phrase insertion like name@section",
    )
    # 出力先を変更したいとき
    p.add_argument(
        "--output-dir",
        "-o",
        help="MIDI 出力ディレクトリを上書き",
    )
    p.add_argument(
        "--output-filename",
        help="MIDI ファイル名 (default: output.mid)",
    )
    p.add_argument("--tempo-curve", help="JSON tempo curve path")
    p.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Number of worker threads for parallel generation",
    )
    p.add_argument(
        "--process-pool",
        action="store_true",
        help="Use multiprocessing instead of threads",
    )
    p.add_argument(
        "--buffer-ahead",
        type=int,
        default=0,
        help="Bars to generate ahead for live output",
    )
    p.add_argument(
        "--parallel-bars",
        type=int,
        default=1,
        help="Number of bars to generate in parallel",
    )
    p.add_argument(
        "--verbose", "-v", action="store_true", help="詳しいログ(INFO)を表示"
    )
    p.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        type=str.upper,
        help="ログレベルを指定",
    )
    p.add_argument("--dry-run", action="store_true", help="動作検証のみ")
    p.add_argument(
        "--strict-drum-map",
        action="store_true",
        help="未知のドラムキーをエラーにする",
    )
    p.add_argument(
        "--expr-curve",
        default="cubic-in",
        help="Expression CC11 curve",
    )
    p.add_argument(
        "--kick-leak-jitter",
        type=int,
        default=0,
        help="Hi-hat velocity jitter range near kicks",
    )
    from utilities.drum_map_registry import DRUM_MAPS

    p.add_argument(
        "--drum-map",
        default="gm",
        choices=DRUM_MAPS.keys(),
        help="使用するドラムマッピングを選択 (default: gm)",
    )
    p.add_argument(
        "--consonant-sync-mode",
        choices=["bar", "note"],
        help="Consonant sync granularity override",
    )
    p.add_argument(
        "--tuning",
        type=_parse_tuning,
        help="Guitar tuning preset name or comma-separated offsets",
    )
    return p


def main_cli() -> None:
    args = build_arg_parser().parse_args()
    configure_logging(args)

    # 1) 設定 & データロード -------------------------------------------------
    main_cfg = load_main_cfg(Path(args.main_cfg))
    if args.strict_drum_map:
        main_cfg.setdefault("global_settings", {})["strict_drum_map"] = True
    if args.drum_map:
        main_cfg.setdefault("global_settings", {})["drum_map"] = args.drum_map
    if args.consonant_sync_mode:
        main_cfg.setdefault("global_settings", {})[
            "consonant_sync_mode"
        ] = args.consonant_sync_mode
    paths = main_cfg.setdefault("paths", {})
    for k, v in (
        ("chordmap_path", args.chordmap),
        ("rhythm_library_path", args.rhythm),
        ("output_dir", args.output_dir),
    ):
        if v:
            paths[k] = v
    if args.time_signature:
        main_cfg.setdefault("global_settings", {})["time_signature"] = args.time_signature
    if args.phrase_spec:
        main_cfg.setdefault("paths", {})["phrase_spec"] = args.phrase_spec
    if args.insert_phrase:
        main_cfg.setdefault("global_settings", {})["insert_phrases"] = args.insert_phrase
    if args.output_filename:
        paths["output_filename"] = args.output_filename
    if args.tempo_curve:
        main_cfg.setdefault("global_settings", {})[
            "tempo_curve_path"
        ] = args.tempo_curve
    if args.tuning:
        pd = main_cfg.setdefault("part_defaults", {})
        for name in ("guitar", "rhythm"):
            if name in pd:
                pd[name]["tuning"] = args.tuning
    tempo_map = None
    curve_path = (
        args.tempo_curve
        or main_cfg.get("global_settings", {}).get("tempo_curve_path")
        or paths.get("tempo_curve_path")
    )
    if curve_path:
        try:
            tempo_map = load_tempo_map(curve_path)
        except Exception as e:  # pragma: no cover - optional
            logger.warning("Failed to load tempo map: %s", e)

    logger.info("使用 chordmap_path = %s", paths["chordmap_path"])
    logger.info("使用 rhythm_library_path = %s", paths["rhythm_library_path"])

    drum_map_name = (
        args.drum_map or main_cfg.get("global_settings", {}).get("drum_map") or "gm"
    )
    main_cfg.setdefault("global_settings", {})["drum_map"] = drum_map_name

    # 3. ファイル読み込み
    chordmap = load_chordmap_yaml(Path(paths["chordmap_path"]))
    rhythm_lib = load_rhythm_library(paths["rhythm_library_path"])

    overrides_model = None
    overrides_path = paths.get("arrangement_overrides_path")
    if overrides_path:
        try:
            from utilities.override_loader import load_overrides  # type: ignore

            overrides_model = load_overrides(overrides_path)
            logger.info("Loaded arrangement overrides from %s", overrides_path)
        except Exception as e:
            logger.error("Failed to load arrangement overrides: %s", e)

    section_names: list[str] = main_cfg["sections_to_generate"]
    raw_sections: dict[str, dict[str, Any]] = chordmap["sections"]
    sections: list[dict[str, Any]] = []
    for name in section_names:
        sec = raw_sections.get(name)
        if not sec:
            logging.warning(f"Section '{name}' not found in chordmap")
            continue
        # ラベルを明示的に保持しておく
        sec_copy = dict(sec)
        sec_copy["label"] = name
        sections.append(sec_copy)

    if not sections:
        logging.error("指定セクションが chordmap に見つかりませんでした")
        return

    ts = meter.TimeSignature(main_cfg["global_settings"].get("time_signature", "4/4"))
    beats_per_measure = ts.numerator

    score, _ = compose(
        main_cfg,
        chordmap,
        rhythm_lib,
        overrides_model=overrides_model,
        tempo_map=tempo_map,
        num_workers=args.threads,
        use_processes=args.process_pool,
        buffer_ahead=args.buffer_ahead,
        parallel_bars=args.parallel_bars,
    )

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
