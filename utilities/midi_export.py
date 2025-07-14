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

try:  # pragma: no cover - optional dependency for export helpers
    import mido
except ImportError:  # pragma: no cover - handled gracefully in functions
    mido = None
import pretty_midi
from music21 import stream
from music21.midi import translate


def write_demo_bar(path: str | Path) -> Path:
    """Write a deterministic 4-beat MIDI bar for tests."""
    out_path = Path(path)
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
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
    track: "mido.MidiTrack",
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


def music21_to_mido(part: stream.Part) -> "mido.MidiFile":
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
    pm: "pretty_midi.PrettyMIDI", tempo_map: list[tuple[float, float]] | None
) -> None:
    """Apply a tempo map to ``pm`` using PrettyMIDI's private API."""
    if tempo_map is None:
        return
    pm._tick_scales = []
    for beat, bpm in tempo_map:
        pm._tick_scales.append(
            (
                int(round(float(beat) * pm.resolution)),
                60.0 / (float(bpm) * pm.resolution),
            )
        )


def export_song(
    bars: int,
    *,
    tempo_map: list[tuple[float, float]] | None = None,
    generators: dict[str, Callable[..., "pretty_midi.PrettyMIDI"]],
    fixed_tempo: float = 120.0,
    out_path: str | Path = "song.mid",
    sections: list[dict[str, Any]] | None = None,
) -> "pretty_midi.PrettyMIDI":
    """Generate multiple parts and export a merged MIDI file."""

    out_path = Path(out_path)
    child_pms: list["pretty_midi.PrettyMIDI"] = []
    if sections:
        for sec in sections:
            vm = vocal_sync.analyse_section(sec, tempo_bpm=fixed_tempo)
            for name, gen in (generators or {}).items():
                pm = gen(sec, fixed_tempo, vocal_metrics=vm)
                apply_tempo_map(pm, tempo_map)
                child_pms.append(pm)
    else:
        for name, gen in (generators or {}).items():
            pm = gen(bars, fixed_tempo)
            apply_tempo_map(pm, tempo_map)
            child_pms.append(pm)

    master = pretty_midi.PrettyMIDI(initial_tempo=fixed_tempo)
    for pm in child_pms:
        for inst in pm.instruments:
            master.instruments.append(copy.deepcopy(inst))

    apply_tempo_map(master, tempo_map)
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


def _import_generator(path: str) -> Callable[[int, float], "pretty_midi.PrettyMIDI"]:
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

    gens: dict[str, Callable[[int, float], "pretty_midi.PrettyMIDI"]] = {}
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
