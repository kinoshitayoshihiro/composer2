from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Iterable, List, Dict

from music21 import converter, harmony
import pretty_midi


def extract_voicings(path: Path) -> Iterator[dict[str, object]]:
    """Yield voicings from ``path`` MIDI file."""
    try:
        score = converter.parse(path)
    except Exception:
        return
    for chord in score.chordify().recurse().getElementsByClass("Chord"):
        if not chord.pitches:
            continue
        try:
            cs = harmony.chordSymbolFromChord(chord)
        except Exception:
            continue
        root = cs.root().name if cs.root() else "C"
        quality = cs.chordKind or cs.figure or "unknown"
        voicing = [int(p.midi) for p in chord.pitches]
        yield {"root": root, "quality": quality, "voicing": voicing}


def load_events(path: Path) -> list[dict[str, object]]:
    """Return note events with basic hand splitting."""

    pm = pretty_midi.PrettyMIDI(str(path))
    tempi, _times = pm.get_tempo_changes()
    tempo = float(tempi[0]) if len(tempi) else 120.0
    if tempo <= 0:
        tempo = 120.0
    beat_dur = 60.0 / tempo
    events: list[dict[str, object]] = []
    for inst in pm.instruments:
        name_upper = inst.name.upper() if inst.name else ""
        inst_hand: str | None = None
        if name_upper.startswith("LH"):
            inst_hand = "lh"
        elif name_upper.startswith("RH"):
            inst_hand = "rh"
        for note in inst.notes:
            start_beat = note.start / beat_dur
            end_beat = note.end / beat_dur
            hand = inst_hand or ("lh" if note.pitch < 60 else "rh")
            events.append(
                {
                    "bar": int(start_beat // 4),
                    "beat": float(start_beat % 4),
                    "dur": float(end_beat - start_beat),
                    "note": int(note.pitch),
                    "velocity": int(note.velocity),
                    "hand": hand,
                }
            )
    events.sort(key=lambda e: (e["bar"], e["beat"]))
    return events


def chunk_events(events: Iterable[dict[str, object]], bars: int = 4) -> Iterator[list[dict[str, object]]]:
    """Yield chunks of ``events`` spanning at most ``bars`` bars."""

    chunk: list[dict[str, object]] = []
    start_bar = None
    for ev in events:
        bar = int(ev["bar"])
        if start_bar is None:
            start_bar = bar
        if bar >= start_bar + bars:
            yield chunk
            chunk = []
            start_bar = bar - (bar % bars)
        chunk.append(ev)
    if chunk:
        yield chunk


def write_corpus(midi_dir: Path, out_path: Path) -> None:
    """Extract events from ``midi_dir`` and write ``out_path`` JSONL."""

    with out_path.open("w", encoding="utf-8") as out_f:
        for midi in sorted(midi_dir.glob("*.mid*")):
            events = load_events(midi)
            for chunk in chunk_events(events):
                json.dump({"file": midi.name, "events": chunk}, out_f, ensure_ascii=False)
                out_f.write("\n")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Extract piano events for corpus")
    parser.add_argument("--midi-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    write_corpus(args.midi_dir, args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
