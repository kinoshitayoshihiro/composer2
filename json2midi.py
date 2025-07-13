"""groove_sampler_v2 JSON を MIDI へ変換するユーティリティ。

複数ファイルを一括処理し、テンポや人間味付けの調整を行えます。

CLI tool to convert groove_sampler_v2 JSON event lists into drum MIDI files.
Supports batch conversion, swing, humanization and YAML mapping options.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import random
import sys
from pathlib import Path

import pretty_midi

__version__ = "0.2.1"

logger = logging.getLogger(__name__)

DEFAULT_PITCH_MAP: dict[str, int] = {
    "kick": 36,
    "snare": 38,
    "chh": 42,
    "hh_edge": 46,
    "hh_pedal": 44,
    "snare_brush": 96,
    "ride": 51,
    "tom_low": 45,
    "tom_mid": 47,
    "tom_high": 50,
}


def _load_json(path: Path) -> dict | list:
    try:
        with path.open() as fh:
            return json.load(fh)
    except FileNotFoundError:
        sys.exit(f"File not found: {path}")
    except json.JSONDecodeError as exc:
        sys.exit(f"JSON parse error in {path}: {exc}")


def _load_mapping(path: Path) -> dict[str, int]:
    ext = path.suffix.lower()
    if ext in {".yaml", ".yml"}:
        try:
            from ruamel.yaml import YAML  # type: ignore
        except Exception:
            sys.exit("ruamel.yaml required for YAML mapping")
        with path.open() as fh:
            data = YAML(typ="safe").load(fh)
    else:
        data = _load_json(path)
    if not isinstance(data, dict):
        sys.exit("Mapping file must define a dictionary")
    return {k: int(v) for k, v in data.items()}


def _beat_to_seconds(
    beat: float, tempo_changes: list[list[float]] | None, bpm: float
) -> float:
    if not tempo_changes:
        return beat * 60.0 / bpm
    tempo_changes = sorted(tempo_changes, key=lambda x: x[0])
    sec = 0.0
    prev_b = 0.0
    prev_t = bpm
    for b, t in tempo_changes:
        if beat < b:
            sec += (beat - prev_b) * 60.0 / prev_t
            return sec
        sec += (b - prev_b) * 60.0 / prev_t
        prev_b, prev_t = b, t
    sec += (beat - prev_b) * 60.0 / prev_t
    return sec


def convert_events(
    events: list[dict[str, float | str]],
    bpm: float,
    mapping: dict[str, int],
    *,
    swing: float = 0.0,
    humanize_timing_ms: float = 0.0,
    humanize_vel_pct: float = 0.0,
    split_tracks: bool = False,
    repeat: int = 1,
    tempo_changes: list[list[float]] | None = None,
    quiet: bool = False,
) -> pretty_midi.PrettyMIDI:
    pattern_len = max(
        float(ev.get("offset", 0)) + float(ev.get("duration", 0)) for ev in events
    )
    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    instruments: dict[str, pretty_midi.Instrument] = {}
    if not split_tracks:
        instruments["drums"] = pretty_midi.Instrument(program=0, is_drum=True)
    warned: set[str] = set()

    total = len(events) * repeat
    bar = None
    if not quiet and total > 100:
        from tqdm import tqdm

        bar = tqdm(total=total, unit="ev", desc="events")

    for rep in range(repeat):
        for ev in events:
            name = str(ev.get("instrument"))
            pitch = mapping.get(name)
            if pitch is None:
                pitch = 35
                if name not in warned:
                    logger.warning("Unknown instrument %s: using pitch 35", name)
                    warned.add(name)
            start_beat = float(ev.get("offset", 0)) + pattern_len * rep
            end_beat = start_beat + float(ev.get("duration", 0))
            if abs(start_beat % 1 - 0.5) < 1e-6:
                shift = swing * 0.25
                start_beat = max(start_beat - shift, 0.0)
                end_beat = max(end_beat - shift, 0.0)
            start = _beat_to_seconds(start_beat, tempo_changes, bpm)
            end = _beat_to_seconds(end_beat, tempo_changes, bpm)
            start += random.uniform(-humanize_timing_ms, humanize_timing_ms) / 1000.0
            velocity = int(min(max(float(ev.get("velocity_factor", 1)) * 127, 1), 127))
            if humanize_vel_pct:
                jitter = random.uniform(-humanize_vel_pct, humanize_vel_pct) / 100.0
                velocity = int(min(max(velocity * (1 + jitter), 1), 127))
            inst = instruments.setdefault(
                name if split_tracks else "drums",
                pretty_midi.Instrument(program=0, is_drum=True),
            )
            inst.notes.append(
                pretty_midi.Note(
                    velocity=velocity, pitch=int(pitch), start=start, end=end
                )
            )
            if bar:
                bar.update(1)
    if bar:
        bar.close()

    pm.instruments.extend(instruments.values())

    if tempo_changes:
        # TODO: replace _tick_scales hack with official API when available
        pm._tick_scales = []
        for beat, tbpm in sorted(tempo_changes, key=lambda x: x[0]):
            tick = int(round(beat * pm.resolution))
            scale = 60.0 / (tbpm * pm.resolution)
            pm._tick_scales.append((tick, scale))
        if pm._tick_scales[0][0] != 0:
            pm._tick_scales.insert(0, (0, 60.0 / (bpm * pm.resolution)))
        max_tick = int(round(pattern_len * repeat * pm.resolution)) + 1
        pm._update_tick_to_time(max_tick)

    return pm


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Convert groove_sampler_v2 JSON to MIDI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("input_json", nargs="+", help="input JSON or glob pattern")
    ap.add_argument("-o", "--out", type=Path, default=None, help="output MIDI path")
    ap.add_argument("-b", "--bpm", type=float, default=120.0, help="tempo in BPM")
    ap.add_argument("--swing", type=float, default=0.0, help="swing amount 0-1")
    ap.add_argument(
        "--humanize-timing", type=float, default=0.0, help="timing jitter ms"
    )
    ap.add_argument(
        "--humanize-vel", type=float, default=0.0, help="velocity jitter %%"
    )
    ap.add_argument(
        "--split-tracks", action="store_true", help="split tracks per instrument"
    )
    ap.add_argument("--repeat", type=int, default=1, help="repeat pattern")
    ap.add_argument("--seed", type=int, default=None, help="random seed")
    ap.add_argument(
        "-m",
        "--map",
        dest="mapping",
        type=Path,
        default=None,
        help="mapping file (JSON/YAML)",
    )
    ap.add_argument(
        "-q", "--quiet", action="store_true", help="suppress progress and summary"
    )
    ap.add_argument("-V", "--version", action="version", version=__version__)
    ns = ap.parse_args(argv)

    swing = max(0.0, min(ns.swing, 1.0))
    if swing != ns.swing:
        logger.error("Invalid swing value: %s", ns.swing)
        sys.exit(1)
    timing = max(0.0, ns.humanize_timing)
    if timing != ns.humanize_timing:
        logger.error("Invalid humanize-timing value: %s", ns.humanize_timing)
        sys.exit(1)
    vel_jitter = max(0.0, ns.humanize_vel)
    if vel_jitter != ns.humanize_vel:
        logger.error("Invalid humanize-vel value: %s", ns.humanize_vel)
        sys.exit(1)

    if ns.seed is not None:
        random.seed(ns.seed)

    inputs: list[Path] = []
    for pattern in ns.input_json:
        matched = list(glob.glob(pattern))
        if not matched:
            sys.exit(f"Input not found: {pattern}")
        inputs.extend(Path(p) for p in matched)

    mapping = DEFAULT_PITCH_MAP
    if ns.mapping is not None:
        mapping = _load_mapping(ns.mapping)

    single_out = ns.out if ns.out and len(inputs) == 1 else None

    for in_path in inputs:
        try:
            data = _load_json(Path(in_path))
            if isinstance(data, dict):
                events = data.get("events", [])
                tempo_changes = data.get("tempo_changes")
                if not isinstance(events, list):
                    raise ValueError("Invalid events")
            else:
                events = data
                tempo_changes = None
            pm = convert_events(
                events,
                ns.bpm,
                mapping,
                swing=swing,
                humanize_timing_ms=timing,
                humanize_vel_pct=vel_jitter,
                split_tracks=ns.split_tracks,
                repeat=ns.repeat,
                tempo_changes=tempo_changes,
                quiet=ns.quiet,
            )
            out_path = (
                single_out if single_out else Path(in_path).with_suffix(".mid")
            )
            pm.write(str(out_path))
            if not ns.quiet:
                msg = f"Saved {out_path} (events: {len(events)}, bpm: {ns.bpm})"
                logger.info(msg)
                print(msg)
        except Exception as exc:
            logger.error("Failed on %s: %s", in_path, exc, exc_info=True)
            sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
