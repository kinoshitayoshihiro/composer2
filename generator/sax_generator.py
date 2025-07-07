from __future__ import annotations

"""Saxophone phrasing generator."""

from typing import Any

from utilities.scale_registry import ScaleRegistry
from utilities.cc_tools import add_cc_events

from music21 import instrument, articulations, spanner, stream
import math

from .melody_generator import MelodyGenerator


DEFAULT_PHRASE_PATTERNS: dict[str, dict[str, Any]] = {
    "sax_basic_swing": {
        "description": "Simple 8th‑note swing phrase (2 bars)",
        "pattern": [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            2.5,
            3.0,
            3.5,
            4.0,
            4.5,
            5.0,
            5.5,
            6.0,
            6.5,
            7.0,
            7.5,
        ],
        "note_duration_ql": 0.5,
        "reference_duration_ql": 8.0,
    },
    "sax_syncopated": {
        "description": "Syncopated phrase with rests",
        "pattern": [0.0, 1.0, 1.5, 2.5, 3.0, 4.0, 4.5, 5.75, 6.25, 7.0],
        "note_duration_ql": 0.5,
        "reference_duration_ql": 8.0,
    },
}

EMOTION_TO_BUCKET: dict[str, str] = {
    "default": "basic",
}

BUCKET_TO_PATTERN: dict[tuple[str, str], str] = {
    ("basic", "low"): "sax_basic_swing",
    ("basic", "medium"): "sax_basic_swing",
    ("basic", "high"): "sax_syncopated",
}

BREATH_CC = 2
MOD_CC = 1
PITCHWHEEL = -1  # stored in extra_cc as negative CC number

STACCATO_VAL = 30
LEGATO_VAL = 90
SLUR_VAL = 80


class SaxGenerator(MelodyGenerator):
    """Melody generator preset for alto saxophone."""

    def __init__(self, **kwargs):
        kwargs.setdefault("instrument_name", "Alto Saxophone")
        rh_lib = kwargs.setdefault("rhythm_library", {})
        for k, v in DEFAULT_PHRASE_PATTERNS.items():
            rh_lib.setdefault(k, v)
        super().__init__(**kwargs)

    # ------------------------------------------------------------------
    # CC Helpers
    # ------------------------------------------------------------------
    def _apply_articulation_cc(self, part: stream.Part) -> None:
        """Add CC1/CC2 events based on articulations."""
        events = []
        notes = sorted(part.recurse().notes, key=lambda n: n.offset)
        prev_end = None
        for n in notes:
            off = float(n.offset)
            is_stacc = any(isinstance(a, articulations.Staccato) for a in n.articulations)
            has_slur = any(isinstance(s, spanner.Slur) and n in s for s in part.recurse().getElementsByClass(spanner.Slur))
            if is_stacc:
                val = STACCATO_VAL
                events.append((off, MOD_CC, val))
            elif has_slur or (prev_end is not None and abs(off - prev_end) < 1e-3):
                val = LEGATO_VAL
                events.append((off, BREATH_CC, val))
            else:
                events.append((off, BREATH_CC, SLUR_VAL))
            prev_end = off + float(n.quarterLength)
        add_cc_events(part, events)

    def _apply_vibrato(self, part: stream.Part, depth: float = 200.0, rate_hz: float = 5.0) -> None:
        """Approximate vibrato using pitch wheel events."""
        events = []
        bpm = float(self.global_tempo or 120.0)
        for n in part.recurse().notes:
            dur_sec = float(n.quarterLength) * 60.0 / bpm
            step = 0.1
            t = 0.0
            while t <= dur_sec + 1e-6:
                val = int(8192 + depth * math.sin(2 * math.pi * rate_hz * t))
                events.append((float(n.offset) + t * bpm / 60.0, PITCHWHEEL, val))
                t += step
        add_cc_events(part, events)

    # ------------------------------------------------------------------
    # Pattern Selection Helpers
    # ------------------------------------------------------------------
    def _choose_pattern_key(
        self,
        emotion: str | None,
        intensity: str | None,
        musical_intent: dict[str, Any] | None = None,
    ) -> str:
        emo = (emotion or "default").lower()
        inten = (intensity or "medium").lower()
        bucket = EMOTION_TO_BUCKET.get(emo, "basic")
        return BUCKET_TO_PATTERN.get((bucket, inten), "sax_basic_swing")

    def compose(self, section_data=None):  # type: ignore[override]
        if section_data:
            mi = section_data.get("musical_intent", {})
            pat_key = self._choose_pattern_key(
                mi.get("emotion"), mi.get("intensity"), mi
            )
            section_data.setdefault("part_params", {}).setdefault("melody", {})[
                "rhythm_key"
            ] = pat_key
        part = super().compose(section_data)
        tonic = (
            section_data.get("tonic_of_section")
            if section_data else self.global_key_signature_tonic
        )
        mode = (
            section_data.get("mode")
            if section_data else self.global_key_signature_mode
        )
        pcs = {
            p.pitchClass
            for p in ScaleRegistry.get(tonic or "C", mode or "major").getPitches(
                "C3", "C5"
            )
        }
        for n in list(part.recurse().notes):
            if n.pitch.pitchClass not in pcs:
                part.remove(n)
        self._apply_articulation_cc(part)
        self._apply_vibrato(part)
        return part
