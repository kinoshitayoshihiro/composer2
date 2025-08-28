from __future__ import annotations

"""Generate a keyswitch staircase MIDI file from a mapping YAML."""

import argparse
import pathlib

from .ujam_map import load_map


def generate(
    product: str,
    out_path: pathlib.Path,
    *,
    note_len: float = 1.0,
    gap: float = 0.1,
    tempo: float = 120.0,
    ppq: int = 480,
    channel: int = 0,
    velocity: int = 100,
) -> None:
    data = load_map(product)
    ks_list = data.get("keyswitches", []) or []
    pairs = [(int(k["note"]), str(k["name"])) for k in ks_list]
    pairs.sort()

    def vlq(val: int) -> bytes:
        out = bytearray([val & 0x7F])
        val >>= 7
        while val:
            out.insert(0, 0x80 | (val & 0x7F))
            val >>= 7
        return bytes(out)

    tempo_us = int(60_000_000 / tempo)
    events = bytearray()
    events += vlq(0) + bytes([0xFF, 0x51, 0x03, (tempo_us >> 16) & 0xFF, (tempo_us >> 8) & 0xFF, tempo_us & 0xFF])
    name_bytes = b"Keyswitches"
    events += vlq(0) + bytes([0xFF, 0x03, len(name_bytes)]) + name_bytes
    note_ticks = int(note_len * ppq)
    gap_ticks = int(gap * ppq)
    first = True
    for pitch, name in pairs:
        delta = 0 if first else gap_ticks
        txt = name.encode("ascii")
        events += vlq(delta) + bytes([0xFF, 0x01, len(txt)]) + txt
        events += vlq(0) + bytes([0x90 + channel, pitch, velocity])
        events += vlq(note_ticks) + bytes([0x80 + channel, pitch, 0])
        first = False
    events += vlq(0) + bytes([0xFF, 0x2F, 0])
    track_len = len(events)
    with open(out_path, "wb") as f:
        f.write(b"MThd")
        f.write((6).to_bytes(4, "big"))
        f.write((0).to_bytes(2, "big"))
        f.write((1).to_bytes(2, "big"))
        f.write(int(ppq).to_bytes(2, "big"))
        f.write(b"MTrk")
        f.write(track_len.to_bytes(4, "big"))
        f.write(events)


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--product", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--note-len", type=float, default=1.0)
    parser.add_argument("--gap", type=float, default=0.1)
    parser.add_argument("--tempo", type=float, default=120.0)
    parser.add_argument("--ppq", type=int, default=480)
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--velocity", type=int, default=100)
    args = parser.parse_args()
    generate(
        args.product,
        pathlib.Path(args.out),
        note_len=args.note_len,
        gap=args.gap,
        tempo=args.tempo,
        ppq=args.ppq,
        channel=args.channel,
        velocity=args.velocity,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
