from __future__ import annotations

from pathlib import Path

import tests._stubs  # noqa: F401
from tools.ujam_bridge import gen_staircase  # type: ignore


def _read_vlq(data: bytes, idx: int) -> tuple[int, int]:
    val = 0
    while True:
        b = data[idx]
        idx += 1
        val = (val << 7) | (b & 0x7F)
        if b < 0x80:
            break
    return val, idx


def _parse_track(data: bytes) -> list[tuple[str, int, int, int]]:
    idx = 0
    time = 0
    events = []
    while idx < len(data):
        delta, idx = _read_vlq(data, idx)
        time += delta
        status = data[idx]
        idx += 1
        if status == 0xFF:
            meta = data[idx]
            idx += 1
            length, idx = _read_vlq(data, idx)
            text = data[idx : idx + length]
            idx += length
            if meta == 0x01:
                events.append(("text", text.decode(), time))
            elif meta == 0x2F:
                break
        else:
            note = data[idx]
            vel = data[idx + 1]
            idx += 2
            chan = status & 0x0F
            typ = "note_on" if status & 0xF0 == 0x90 and vel > 0 else "note_off"
            events.append((typ, note, chan, time))
    return events


def test_gen_staircase_basic(tmp_path: Path) -> None:
    out = tmp_path / "out.mid"
    gen_staircase.generate(
        "iron2",
        out,
        note_len=1.0,
        gap=0.5,
        tempo=120.0,
        ppq=480,
        channel=2,
        velocity=90,
    )
    data = (out.read_bytes())[22:]  # skip header 14 + track header 8
    events = _parse_track(data)
    texts = [e[1] for e in events if e[0] == "text"]
    note_events = [e for e in events if e[0] in {"note_on", "note_off"}]
    notes_order = [note_events[i][1] for i in range(0, len(note_events), 2)]
    durations = [
        note_events[i + 1][3] - note_events[i][3]
        for i in range(0, len(note_events), 2)
    ]
    channels = [note_events[i][2] for i in range(0, len(note_events), 2)]
    assert notes_order == [20, 22, 24]
    assert all(d == 480 for d in durations)
    assert all(ch == 2 for ch in channels)
    assert texts == ["a", "b", "c"]
