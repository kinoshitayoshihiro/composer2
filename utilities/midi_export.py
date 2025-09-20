from __future__ import annotations

import copy
import csv
import importlib
import io
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from utilities import vocal_sync
from utilities.midi_utils import safe_end_time

try:  # optional dependency introduced by tempo merging
    from utilities import tempo_utils
except Exception as exc:  # pragma: no cover - missing core extras
    raise RuntimeError("tempo_utils missing — run pip install .[core]") from exc

try:  # pragma: no cover - optional dependency for export helpers
    import mido
except ImportError:  # pragma: no cover - handled gracefully in functions
    mido = None
import pretty_midi
from music21 import stream
from music21.midi import translate

from utilities.pretty_midi_safe import new_pm as PrettyMIDI


def _set_initial_tempo(pm: "pretty_midi.PrettyMIDI", bpm: float) -> None:
    """Set the base tempo of ``pm`` using PrettyMIDI's internal tick scales."""

    scale = 60.0 / (float(bpm) * pm.resolution)
    pm._tick_scales = [(0, scale)]
    pm._update_tick_to_time(pm.resolution)


def write_demo_bar(path: str | Path) -> Path:
    """Write a deterministic 4-beat MIDI bar for tests."""
    out_path = Path(path)
    pm = PrettyMIDI()
    _set_initial_tempo(pm, 120.0)
    inst = pretty_midi.Instrument(program=0)
    qlen = 0.5  # quarter note length in seconds at 120 BPM
    for i in range(4):
        start = i * qlen
        note = pretty_midi.Note(velocity=90, pitch=60, start=start, end=start + qlen)
        inst.notes.append(note)
    pm.instruments.append(inst)
    pm.write(str(out_path))
    return out_path


def append_extra_cc(
    part: stream.Part,
    track: mido.MidiTrack,
    to_ticks: Callable[[float], int],
    *,
    channel: int = 0,
) -> None:
    """Append extra CC events from ``part`` to the given ``track``."""
    if mido is None:
        return
    if hasattr(part, "extra_cc"):
        for cc in part.extra_cc:
            track.append(
                mido.Message(
                    "control_change",
                    time=to_ticks(cc["time"]),
                    control=int(cc["cc"]),
                    value=int(cc["val"]),
                    channel=channel,
                )
            )


def music21_to_mido(part: stream.Part) -> mido.MidiFile:
    """Convert a ``music21`` part to ``mido.MidiFile`` preserving ``extra_cc``."""
    if mido is None:  # pragma: no cover - optional dependency
        raise ImportError("mido is required to export MIDI")

    mf = translate.streamToMidiFile(part)
    data = mf.writestr()
    midi = mido.MidiFile(file=io.BytesIO(data))

    def to_ticks(ql: float) -> int:
        return int(round(ql * midi.ticks_per_beat))

    if midi.tracks:
        append_extra_cc(part, midi.tracks[0], to_ticks)
    return midi


def apply_tempo_map(
    pm: PrettyMIDI, tempo_map: list[tuple[float, float]] | None
) -> None:
    """Apply a tempo map to ``pm`` using PrettyMIDI's public API."""
    if tempo_map is None:
        return
    tempo_map = sorted((float(b), float(t)) for b, t in tempo_map)
    tempi = [int(round(bpm)) for _, bpm in tempo_map]
    times = [tempo_utils.beat_to_seconds(b, tempo_map) for b, _ in tempo_map]

    # PrettyMIDI has no public API to set tempo changes. We directly update
    # ``_tick_scales`` which ``set_tempo_changes`` would normally modify.
    if tempi:
        pm._tick_scales = []
        tick = 0.0
        last_time = times[0]
        last_scale = 60.0 / (float(tempi[0]) * pm.resolution)
        for i, (bpm, start) in enumerate(zip(tempi, times)):
            if i > 0:
                tick += (start - last_time) / last_scale
                last_scale = 60.0 / (float(bpm) * pm.resolution)
                last_time = start
            pm._tick_scales.append((int(round(tick)), last_scale))
        pm._update_tick_to_time(int(round(tick)) + 1)


def export_song(
    bars: int,
    *,
    tempo_map: list[tuple[float, float]] | None = None,
    generators: dict[str, Callable[..., PrettyMIDI]],
    fixed_tempo: float = 120.0,
    out_path: str | Path = "song.mid",
    sections: list[dict[str, Any]] | None = None,
) -> PrettyMIDI:
    """Generate multiple parts and export a merged MIDI file.

    When ``sections`` are provided each section may include a ``tempo_map`` key
    containing ``[(beat, bpm), ...]`` pairs. Generated parts for that section are
    offset by the cumulative beat position and all tempo change events are
    applied once to the master ``PrettyMIDI`` object.
    """

    out_path = Path(out_path)
    base_tempo = float(fixed_tempo or 120.0)
    master = PrettyMIDI(initial_tempo=base_tempo)
    _set_initial_tempo(master, base_tempo)
    all_tempos: list[tuple[float, float]] = []
    tempo_tuple: tuple[tuple[float, float], ...] | None = None

    if sections:
        beat_offset = 0.0
        sec_per_beat = 60.0 / base_tempo

        for sec in sections:
            vm = vocal_sync.analyse_section(sec, tempo_bpm=base_tempo)
            sec_pms: list[PrettyMIDI] = []
            for name, gen in (generators or {}).items():
                sec_pms.append(gen(sec, base_tempo, vocal_metrics=vm))

            # determine section duration from generated parts
            sec_duration = 0.0
            for pm in sec_pms:
                sec_duration = max(sec_duration, safe_end_time(pm))

            sec_tm = sec.get("tempo_map") or []
            new_tempo = False
            for beat, bpm in sec_tm:
                all_tempos.append((beat + beat_offset, bpm))
                new_tempo = True
            if tempo_tuple is None or new_tempo:
                tempo_tuple = tuple(all_tempos)
                new_tempo = False

            for pm in sec_pms:
                for inst in pm.instruments:
                    inst_copy = copy.deepcopy(inst)
                    for note in inst_copy.notes:
                        start_b = beat_offset + note.start / sec_per_beat
                        end_b = beat_offset + note.end / sec_per_beat
                        note.start = tempo_utils.beat_to_seconds(start_b, tempo_tuple)
                        note.end = tempo_utils.beat_to_seconds(end_b, tempo_tuple)
                    for cc in inst_copy.control_changes:
                        b = beat_offset + cc.time / sec_per_beat
                        cc.time = tempo_utils.beat_to_seconds(b, tempo_tuple)
                    for bend in inst_copy.pitch_bends:
                        b = beat_offset + bend.time / sec_per_beat
                        bend.time = tempo_utils.beat_to_seconds(b, tempo_tuple)
                    master.instruments.append(inst_copy)

            if "bars" in sec:
                sec_beats = float(sec["bars"]) * 4.0
            else:
                sec_beats = sec_duration / sec_per_beat
            beat_offset += sec_beats
    else:
        for name, gen in (generators or {}).items():
            pm = gen(bars, base_tempo)
            for inst in pm.instruments:
                master.instruments.append(copy.deepcopy(inst))
        all_tempos = list(tempo_map or [])
        tempo_tuple = None

    if all_tempos or tempo_map:
        # fallback to provided tempo_map when no sections were used
        if tempo_map and not sections:
            all_tempos = list(tempo_map)
        all_tempos.sort(key=lambda x: float(x[0]))
        if all_tempos:
            apply_tempo_map(master, all_tempos)

    master.write(str(out_path))
    return master


def _load_tempo_map(path: Path) -> list[tuple[float, float]]:
    if path.suffix.lower() == ".csv":
        with path.open() as fh:
            reader = csv.reader(fh)
            return [(float(b), float(t)) for b, t in reader]
    with path.open() as fh:
        data = json.load(fh)
    return [(float(b), float(t)) for b, t in data]


def _import_generator(path: str) -> Callable[[int, float], PrettyMIDI]:
    mod = importlib.import_module(path)
    return getattr(mod, "generate")


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Export MIDI song")
    parser.add_argument("bars", type=int)
    parser.add_argument("--tempo-map", dest="tempo_map")
    parser.add_argument("--out", default="song.mid")
    parser.add_argument("--fixed-t", type=float, default=120.0)
    parser.add_argument(
        "--drum",
        default="generator.drum_generator",
        help="import path to drum generator",
    )
    parser.add_argument("--bass")
    parser.add_argument("--piano")

    args = parser.parse_args(argv)

    gens: dict[str, Callable[[int, float], PrettyMIDI]] = {}
    if args.drum:
        gens["drum"] = _import_generator(args.drum)
    if args.bass:
        gens["bass"] = _import_generator(args.bass)
    if args.piano:
        gens["piano"] = _import_generator(args.piano)

    tempo = None
    if args.tempo_map:
        tempo = _load_tempo_map(Path(args.tempo_map))

    export_song(
        args.bars,
        tempo_map=tempo,
        generators=gens,
        fixed_tempo=args.fixed_t,
        out_path=args.out,
    )


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    main()


__all__ = ["export_song", "apply_tempo_map"]
