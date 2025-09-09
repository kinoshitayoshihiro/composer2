from __future__ import annotations

"""Generate a key switch YAML template from a MIDI "staircase" file."""

import argparse
from typing import Dict

import pretty_midi
import yaml


def learn_keymap(in_midi: str, out_yaml: str) -> None:
    pm = pretty_midi.PrettyMIDI(in_midi)
    pitches = sorted({n.pitch for inst in pm.instruments for n in inst.notes})
    if not pitches:
        raise ValueError("No notes found in MIDI")
    mapping: Dict[str, int] = {f"ks_{i}": p for i, p in enumerate(pitches)}
    data = {
        "plugin": "unknown",
        "play_range": {"low": min(pitches), "high": max(pitches)},
        "keyswitch": mapping,
    }
    with open(out_yaml, "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="in_midi", required=True)
    parser.add_argument("--out", dest="out_yaml", required=True)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    learn_keymap(args.in_midi, args.out_yaml)


if __name__ == "__main__":  # pragma: no cover
    main()
