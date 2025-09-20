from __future__ import annotations

import pretty_midi


def safe_end_time(pm: pretty_midi.PrettyMIDI) -> float:
    """Return the maximum timestamp from all events in *pm*.

    This manually traverses instruments, meta events and tempo changes
    without assuming that ``pm.get_tempo_changes()`` returns numpy arrays.
    """
    end = 0.0
    # instrument events
    for inst in pm.instruments:
        for note in inst.notes:
            if note.end > end:
                end = note.end
        for cc in inst.control_changes:
            if cc.time > end:
                end = cc.time
        for pb in inst.pitch_bends:
            if pb.time > end:
                end = pb.time
    # meta events
    for attr in (
        "lyrics",
        "text_events",
        "time_signature_changes",
        "key_signature_changes",
    ):
        for ev in getattr(pm, attr, []):
            t = getattr(ev, "time", 0.0)
            if t > end:
                end = float(t)
    # tempo changes
    _, times = pm.get_tempo_changes()
    for t in times:
        if float(t) > end:
            end = float(t)
    return float(end)


__all__ = ["safe_end_time"]
