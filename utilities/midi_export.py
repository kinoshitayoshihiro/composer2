from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import io

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

