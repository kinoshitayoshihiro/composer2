"""Helpers for light-weight MIDI note editing used across generators."""
from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List

try:  # pragma: no cover - pretty_midi is an optional dependency in some envs
    import pretty_midi
except Exception:  # pragma: no cover
    pretty_midi = None  # type: ignore

if TYPE_CHECKING:  # pragma: no cover
    import pretty_midi as _pm


def _require_pretty_midi() -> None:
    if pretty_midi is None:  # pragma: no cover - defensive
        raise ImportError("pretty_midi is required for utilities.midi_edit")


def dedupe_stack(inst: "pretty_midi.Instrument", *, min_sep_sec: float = 0.005) -> None:
    """Remove near-identical stacked notes.

    Notes sharing the same pitch and starting within ``min_sep_sec`` are considered
    duplicates. The longer / louder note is kept to preserve phrasing.
    """

    _require_pretty_midi()
    if not inst.notes:
        return

    inst.notes.sort(key=lambda n: (n.pitch, n.start, n.end, n.velocity))
    filtered: List[pretty_midi.Note] = []
    last: pretty_midi.Note | None = None
    for note in inst.notes:
        if last and note.pitch == last.pitch:
            same_onset = abs(note.start - last.start) <= min_sep_sec
            overlapping = note.start < last.end and (note.start - last.start) <= min_sep_sec
            if same_onset or overlapping:
                keep_current = (
                    (note.end - note.start, note.velocity)
                    > (last.end - last.start, last.velocity)
                )
                if keep_current:
                    last.start = min(last.start, note.start)
                    last.end = max(last.end, note.end)
                    last.velocity = max(last.velocity, note.velocity)
                continue
        filtered.append(note)
        last = note

    filtered.sort(key=lambda n: (n.start, n.pitch, n.end))
    inst.notes = filtered


def merge_ties(inst: "pretty_midi.Instrument", *, gap_sec: float = 0.005) -> None:
    """Merge back-to-back notes of the same pitch separated by a small gap."""

    _require_pretty_midi()
    if not inst.notes:
        return

    inst.notes.sort(key=lambda n: (n.pitch, n.start, n.end))
    merged: List[pretty_midi.Note] = []
    for note in inst.notes:
        if (
            merged
            and note.pitch == merged[-1].pitch
            and note.start - merged[-1].end <= gap_sec
            and note.start >= merged[-1].start
        ):
            merged[-1].end = max(merged[-1].end, note.end)
            merged[-1].velocity = max(merged[-1].velocity, note.velocity)
        else:
            merged.append(note)
    inst.notes = merged


def hold_once_per_bar(inst: "pretty_midi.Instrument", *, bar_len_sec: float) -> None:
    """Collapse notes into one sustained hold per bar for each pitch."""

    _require_pretty_midi()
    if bar_len_sec <= 0 or not inst.notes:
        return

    grouped: Dict[int, Dict[int, pretty_midi.Note]] = defaultdict(dict)
    for note in inst.notes:
        bar_idx = int(note.start // bar_len_sec)
        slot = grouped[bar_idx].get(note.pitch)
        if slot is None or (note.velocity, note.end - note.start) > (
            slot.velocity,
            slot.end - slot.start,
        ):
            grouped[bar_idx][note.pitch] = note

    new_notes: List[pretty_midi.Note] = []
    for bar_idx in sorted(grouped):
        bar_start = bar_idx * bar_len_sec
        bar_end = bar_start + bar_len_sec
        for pitch, ref in grouped[bar_idx].items():
            new_notes.append(
                pretty_midi.Note(
                    velocity=ref.velocity,
                    pitch=pitch,
                    start=bar_start,
                    end=bar_end,
                )
            )

    new_notes.sort(key=lambda n: (n.start, n.pitch))
    inst.notes = new_notes


def to_trigger_per_bar(
    inst: "pretty_midi.Instrument", *, bar_len_sec: float, trigger_pitch: int
) -> None:
    """Convert each bar to a short trigger note."""

    _require_pretty_midi()
    if bar_len_sec <= 0 or not inst.notes:
        return

    grouped: Dict[int, List[pretty_midi.Note]] = defaultdict(list)
    for note in inst.notes:
        bar_idx = int(note.start // bar_len_sec)
        grouped[bar_idx].append(note)

    new_notes: List[pretty_midi.Note] = []
    pulse = max(0.02, min(0.25 * bar_len_sec, 0.10))
    for bar_idx in sorted(grouped):
        if not grouped[bar_idx]:
            continue
        bar_start = bar_idx * bar_len_sec
        vel = max(n.velocity for n in grouped[bar_idx])
        new_notes.append(
            pretty_midi.Note(
                velocity=vel,
                pitch=int(trigger_pitch),
                start=bar_start,
                end=bar_start + pulse,
            )
        )

    new_notes.sort(key=lambda n: n.start)
    inst.notes = new_notes


def light_cleanup(inst: "pretty_midi.Instrument") -> None:
    """Lightweight cleanup used for performance exports.

    - Remove duplicated stacks within 5ms.
    - Merge near-contiguous notes of the same pitch when the gap is <=10ms.
    """

    dedupe_stack(inst, min_sep_sec=0.005)
    merge_ties(inst, gap_sec=0.010)
