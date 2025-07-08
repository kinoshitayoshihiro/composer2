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

    def __init__(
        self,
        seed: int | None = None,
        *,
        staccato_prob: float = 0.3,
        slur_prob: float = 0.5,
        vibrato_depth: float = 200.0,
        vibrato_rate: float = 5.0,
        **kwargs,
    ):
        kwargs.setdefault("instrument_name", "Alto Saxophone")
        kwargs["default_instrument"] = instrument.AltoSaxophone()
        rh_lib = kwargs.setdefault("rhythm_library", {})
        for k, v in DEFAULT_PHRASE_PATTERNS.items():
            rh_lib.setdefault(k, v)

        if "rng" not in kwargs:
            kwargs["rng"] = random.Random(seed)

        super().__init__(**kwargs)
        self.staccato_prob = float(staccato_prob)
        self.slur_prob = float(slur_prob)
        self.vibrato_depth = float(vibrato_depth)
        self.vibrato_rate = float(vibrato_rate)

    # ------------------------------------------------------------------
    # CC Helpers
    # ------------------------------------------------------------------
    def _apply_articulation_cc(self, part: stream.Part) -> None:
        """Add CC1/CC2 events based on articulations."""
        events: list[tuple[float, int, int]] = []
        notes = sorted(part.recurse().notes, key=lambda n: float(n.offset))

        # collect note ids belonging to slur spanners
        slur_notes: set[int] = set()
        for sl in part.recurse().getElementsByClass(spanner.Slur):
            for el in sl.getSpannedElements():
                if isinstance(el, note.Note):
                    slur_notes.add(id(el))

        prev_end: float | None = None
        for n in notes:
            off = float(n.offset)

            is_stacc = any(isinstance(a, articulations.Staccato) for a in n.articulations)
            in_slur = id(n) in slur_notes or any(
                isinstance(a, articulations.Tenuto) for a in n.articulations
            )
            near_prev_end = prev_end is not None and abs(off - prev_end) < 1e-3

            if is_stacc:
                events.append((off, MOD_CC, STACCATO_VAL))
            elif in_slur or near_prev_end:
                events.append((off, BREATH_CC, LEGATO_VAL))
            else:
                events.append((off, BREATH_CC, SLUR_VAL))

            prev_end = off + float(n.quarterLength)

        # merge events and remove duplicates using cc_tools
        add_cc_events(part, events)

    def _apply_vibrato(
        self,
        part: stream.Part,
        depth: float | None = None,
        rate_hz: float | None = None,
        *,
        step_ql: float = 0.1,
    ) -> None:
        """Approximate vibrato using pitch wheel events.

        Parameters
        ----------
        part:
            Target part.
        depth:
            Pitch wheel offset in integer units. ``None`` uses
            :attr:`vibrato_depth`.
        rate_hz:
            Vibrato rate in Hertz. ``None`` uses :attr:`vibrato_rate`.
        step_ql:
            Step width in quarterLength for waveform generation.
        """

        depth = self.vibrato_depth if depth is None else depth
        rate_hz = self.vibrato_rate if rate_hz is None else rate_hz

        events: list[tuple[float, int, int]] = []
        bpm = float(self.global_tempo or 120.0)
        for n in part.recurse().notes:
            dur_ql = float(n.quarterLength)
            t = 0.0
            while t <= dur_ql + 1e-6:
                sec = t * 60.0 / bpm
                raw = 8192 + depth * math.sin(2 * math.pi * rate_hz * sec)
                val = max(0, min(16383, int(raw)))
                events.append((float(n.offset) + t, PITCHWHEEL, val))
                t += step_ql

        # merge and sort events
        add_cc_events(part, events)

    def _apply_velocity_curve(self, part: stream.Part, intensity: str) -> None:
        """Add CC11 dynamics across the phrase based on intensity."""
        curve_map = {
            "low": (50, 70),
            "medium": (60, 90),
            "high": (80, 110),
        }
        start_val, end_val = curve_map.get(intensity.lower(), (60, 90))

        notes = sorted(part.recurse().notes, key=lambda n: float(n.offset))
        if not notes:
            return

        events: list[tuple[float, int, int]] = []
        for idx, n in enumerate(notes):
            frac = idx / max(1, len(notes) - 1)
            val = int(round(start_val + (end_val - start_val) * frac))
            events.append((float(n.offset), 11, val))

        dedup: list[tuple[float, int, int]] = []
        prev: tuple[int, int] | None = None
        for t, c, v in events:
            if prev is None or prev != (c, v):
                dedup.append((t, c, v))
            prev = (c, v)

        add_cc_events(part, dedup)

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

        prev_note: note.Note | None = None
        for off in pat.get("pattern", []):
            if not scale_pitches:
                continue
            pitch_obj = self.rng.choice(scale_pitches)
            n = note.Note(pitch_obj)
            n.quarterLength = pat.get("note_duration_ql", 0.5)
            n.volume.velocity = 90
            part.insert(float(off), n)

            if self.rng.random() < self.staccato_prob:
                n.articulations.append(articulations.Staccato())
            elif prev_note is not None and self.rng.random() < self.slur_prob:
                sl = spanner.Slur([prev_note, n])
                part.insert(float(prev_note.offset), sl)

            prev_note = n

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
        intensity = section_data.get("musical_intent", {}).get("intensity", "medium") if section_data else "medium"
        self._apply_articulation_cc(part)
        self._apply_vibrato(part)
        self._apply_velocity_curve(part, intensity)
        return part
