from __future__ import annotations

from pathlib import Path
import shutil
from typing import List, Tuple

import pretty_midi


def _extract_events(pm: pretty_midi.PrettyMIDI) -> List[Tuple[int, bool, int, float, float, int]]:
    """Return a normalised list of note events for comparison."""

    events: List[Tuple[int, bool, int, float, float, int]] = []
    for inst in pm.instruments:
        for n in inst.notes:
            events.append(
                (
                    inst.program,
                    inst.is_drum,
                    n.pitch,
                    round(n.start, 6),
                    round(n.end, 6),
                    n.velocity,
                )
            )
    events.sort()
    return events


def compare_midi(original: Path, generated: Path) -> bool:
    """Return True if MIDI files contain the same note events."""

    try:
        pm_orig = pretty_midi.PrettyMIDI(str(original))
        pm_gen = pretty_midi.PrettyMIDI(str(generated))
    except Exception:  # pragma: no cover - parse errors
        return False

    return _extract_events(pm_orig) == _extract_events(pm_gen)


def update_golden(src: Path, dst: Path) -> None:
    """Replace ``dst`` with ``src`` preserving file permissions."""
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    shutil.copy2(src, tmp)
    tmp.replace(dst)
