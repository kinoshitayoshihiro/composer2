from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PeakSyncConfig:
    lag_ms: float = 10.0
    min_distance_beats: float = 0.25
    sustain_threshold_ms: float = 120.0


class PeakSynchroniser:
    """Synchronise drum events with consonant peaks."""

    TICK_RESOLUTION = 960

    @staticmethod
    def _quantize(off: float, step: float = 1e-3) -> float:
        """offset を粗いグリッドに丸めて衝突検出に使う。"""
        return round(off / step) * step

    @staticmethod
    def _add_event(
        events: list[dict[str, float | str]],
        instrument: str,
        offset: float,
        *,
        priority: dict[str, int],
    ) -> None:
        q_off = PeakSynchroniser._quantize(offset)
        for idx, ev in enumerate(events):
            ev_off = PeakSynchroniser._quantize(float(ev.get("offset", 0.0)))
            if abs(ev_off - q_off) <= 1e-6:
                if priority.get(instrument, 0) > priority.get(ev.get("instrument", ""), 0):
                    events[idx] = {
                        "instrument": instrument,
                        "offset": q_off,
                        "duration": ev.get("duration", 0.25),
                    }
                return
        events.append({"instrument": instrument, "offset": q_off, "duration": 0.25})

    @staticmethod
    def sync_events(
        peaks: list[float],
        base_events: list[dict[str, float | str]],
        *,
        tempo_bpm: float,
        lag_ms: float = 10.0,
        min_distance_beats: float = 0.25,
        sustain_threshold_ms: float = 120.0,
        clip_at_zero: bool = False,
    ) -> list[dict[str, float | str]]:
        """Return ``base_events`` augmented with hits aligned to ``peaks``.

        ``clip_at_zero`` ensures negative offsets caused by a negative lag are
        clipped to ``0.0`` beats rather than producing pre-bar events.
        """
        sec_per_beat = 60.0 / tempo_bpm
        lag_beats = (lag_ms / 1000.0) / sec_per_beat
        priority = {"kick": 3, "snare": 2, "ohh": 1}
        events = [dict(ev) for ev in base_events]
        peaks_sorted = sorted(float(p) for p in peaks)
        last_beat = -1e9
        for idx, p in enumerate(peaks_sorted):
            beat = p / sec_per_beat
            if beat - last_beat < min_distance_beats:
                continue
            last_beat = beat
            quant = round(beat * 2) / 2
            final_off = quant + lag_beats
            if clip_at_zero and final_off < 0:
                final_off = 0.0
            final_off = PeakSynchroniser._quantize(final_off)
            if abs(quant - round(quant)) < 1e-6:
                instr = "kick" if int(round(quant)) % 2 == 0 else "snare"
            else:
                instr = "snare"
            PeakSynchroniser._add_event(events, instr, final_off, priority=priority)
            next_gap = (
                peaks_sorted[idx + 1] - p if idx + 1 < len(peaks_sorted) else float("inf")
            )
            if next_gap * 1000 >= sustain_threshold_ms:
                PeakSynchroniser._add_event(events, "ohh", final_off, priority=priority)
                pedal_off = final_off + 0.25
                PeakSynchroniser._add_event(events, "hh_pedal", pedal_off, priority=priority)
        events.sort(key=lambda e: float(e.get("offset", 0.0)))
        return events
