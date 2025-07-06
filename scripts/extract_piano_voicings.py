from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from music21 import converter, harmony


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


def main() -> None:
    corpus = Path("data/piano_corpus")
    out_path = Path("piano_voicings.jsonl")
    with out_path.open("w", encoding="utf-8") as out_f:
        for midi in sorted(corpus.glob("*.mid*")):
            for item in extract_voicings(midi):
                json.dump(item, out_f, ensure_ascii=False)
                out_f.write("\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
