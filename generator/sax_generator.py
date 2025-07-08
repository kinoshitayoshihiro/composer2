from __future__ import annotations

"""Saxophone phrasing generator."""

from typing import Any

from utilities.scale_registry import ScaleRegistry
from utilities.cc_tools import add_cc_events
import random

from music21 import instrument, articulations, spanner, stream, note
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

    def __init__(self, seed: int | None = None, **kwargs):
        kwargs.setdefault("instrument_name", "Alto Saxophone")
        kwargs["default_instrument"] = instrument.AltoSaxophone()
        rh_lib = kwargs.setdefault("rhythm_library", {})
        for k, v in DEFAULT_PHRASE_PATTERNS.items():
            rh_lib.setdefault(k, v)

        if "rng" not in kwargs:
            kwargs["rng"] = random.Random(seed)

        super().__init__(**kwargs)

    # ------------------------------------------------------------------
    # CC Helpers
    # ------------------------------------------------------------------
    def _apply_articulation_cc(self, part: stream.Part) -> None:
        """Add CC1/CC2 events based on articulations."""
        events: list[tuple[float, int, int]] = []
        notes = list(part.recurse().notes)
        slurs = list(part.recurse().getElementsByClass(spanner.Slur))
        prev_end: float | None = None
        for n in notes:
            off = float(n.offset)
            is_stacc = any(isinstance(a, articulations.Staccato) for a in n.articulations)
            in_slur = any(n in s for s in slurs)
            near_prev_end = prev_end is not None and abs(off - prev_end) < 1e-3

            if is_stacc:
                events.append((off, MOD_CC, STACCATO_VAL))
            elif in_slur or near_prev_end:
                events.append((off, BREATH_CC, LEGATO_VAL))
            else:
                events.append((off, BREATH_CC, SLUR_VAL))

            prev_end = off + float(n.quarterLength)

        add_cc_events(part, events)
        part.extra_cc = [{"time": t, "cc": c, "value": v} for t, c, v in events]

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

    def _render_part(
        self,
        section_data: dict[str, Any],
        next_section_data: dict[str, Any] | None = None,
    ) -> stream.Part:
        pattern_key = (
            section_data.get("part_params", {})
            .get("melody", {})
            .get("rhythm_key", "sax_basic_swing")
        )
        pat = DEFAULT_PHRASE_PATTERNS.get(pattern_key, DEFAULT_PHRASE_PATTERNS["sax_basic_swing"])

        tonic = section_data.get("tonic_of_section", self.global_key_signature_tonic)
        mode = section_data.get("mode", self.global_key_signature_mode)
        scale_pitches = ScaleRegistry.get(tonic or "C", mode or "major").getPitches("C3", "C5")

        part = stream.Part(id=self.part_name or "sax")
        part.insert(0, self.default_instrument)

        for off in pat.get("pattern", []):
            if not scale_pitches:
                continue
            pitch_obj = self.rng.choice(scale_pitches)
            n = note.Note(pitch_obj)
            n.quarterLength = pat.get("note_duration_ql", 0.5)
            n.volume.velocity = 90
            part.insert(float(off), n)

        return part

    def compose(self, section_data=None):  # type: ignore[override]
        if section_data:
            mi = section_data.get("musical_intent", {})
            pat_key = self._choose_pattern_key(
                mi.get("emotion"), mi.get("intensity"), mi
            )
            section_data.setdefault("part_params", {}).setdefault("melody", {})[
                "rhythm_key"
            ] = pat_key
        part = self._render_part(section_data)
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
