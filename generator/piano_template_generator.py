from __future__ import annotations

import copy
from typing import Any

from music21 import chord, expressions, harmony, note, stream, volume

from utilities import humanizer
from utilities.humanizer import apply_swing
from utilities.cc_tools import merge_cc_events
from utilities.pedalizer import generate_pedal_cc

from .voicing_density import VoicingDensityEngine

from .base_part_generator import BasePartGenerator

PPQ = 480

class PianoTemplateGenerator(BasePartGenerator):
    """Very simple piano generator for alpha testing."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._density_engine = VoicingDensityEngine()

    def _render_part(
        self, section_data: dict[str, Any], next_section_data: dict[str, Any] | None = None
    ) -> dict[str, stream.Part]:
        chord_label = section_data.get("chord_symbol_for_voicing", "C")
        groove_kicks: list[float] = section_data.get("groove_kicks", [])
        musical_intent = section_data.get("musical_intent", {})
        intensity = musical_intent.get("intensity", "medium")
        voicing_mode = section_data.get("voicing_mode", "shell")
        base_vel = {"low": 60, "medium": 70, "high": 80}.get(str(intensity), 70)

        try:
            cs = harmony.ChordSymbol(chord_label)
            cs.closedPosition(inPlace=True)
        except Exception:
            cs = harmony.ChordSymbol("C")

        root = cs.root() or harmony.ChordSymbol("C").root()
        shell: list[harmony.ChordSymbol] = []
        if voicing_mode != "guide":
            shell.append(root)
        if cs.third:
            shell.append(cs.third)
        if cs.seventh:
            shell.append(cs.seventh)
        elif cs.fifth:
            shell.append(cs.fifth)
        if voicing_mode == "drop2" and len(shell) >= 2:
            shell = shell[:-2] + [shell[-2].transpose(-12)] + [shell[-1]]

        q_length = float(section_data.get("q_length", self.bar_length))

        rh = stream.Part(id="piano_rh")
        lh = stream.Part(id="piano_lh")
        rh.insert(0, copy.deepcopy(self.default_instrument))
        lh.insert(0, copy.deepcopy(self.default_instrument))

        # Right hand: shell chords in eighth notes
        eight = 0.5
        off = 0.0
        while off < q_length:
            c = chord.Chord(shell, quarterLength=min(eight, q_length - off))
            c.volume = volume.Volume(velocity=base_vel)
            rh.insert(off, c)
            off += eight

        # Left hand: root notes in quarter notes
        quarter = 1.0
        off = 0.0
        while off < q_length:
            n = note.Note(root, quarterLength=min(quarter, q_length - off))
            n.volume = volume.Volume(velocity=base_vel)
            lh.insert(off, n)
            off += quarter

        # Kick lock velocity boost
        if groove_kicks:
            eps = 3 / PPQ
            for part in (rh, lh):
                for n in part.recurse().notes:
                    if any(abs(float(n.offset) - k) <= eps for k in groove_kicks):
                        n.volume = n.volume or volume.Volume(velocity=base_vel)
                        n.volume.velocity = min(127, int(n.volume.velocity) + 10)

        chord_stream = section_data.get("chord_stream")
        if chord_stream:
            events = generate_pedal_cc(chord_stream)
            for part in (rh, lh):
                base: set[tuple[float, int, int]] = set(getattr(part, "_extra_cc", set()))
                part._extra_cc = set(merge_cc_events(base, events))

        if section_data.get("use_pedal"):
            cc_events = self._pedal_marks(intensity, q_length)
            for ev in cc_events:
                ped = expressions.PedalMark()
                if hasattr(ped, "pedalType"):
                    ped.pedalType = expressions.PedalType.Sustain
                rh.insert(ev["time"], ped)
                lh.insert(ev["time"], copy.deepcopy(ped))
            for part in (rh, lh):
                part.extra_cc = getattr(part, "extra_cc", []) + cc_events
        profile = (
            section_data.get("humanize_profile")
            or self.global_settings.get("humanize_profile")
        )
        swing_ratio = (
            section_data.get("swing_ratio")
            or self.global_settings.get("swing_ratio")
        )
        for p in (rh, lh):
            if profile:
                humanizer.apply(p, profile)
            if swing_ratio:
                apply_swing(p, float(swing_ratio), subdiv=8)

        return {"piano_rh": rh, "piano_lh": lh}

    def _pedal_marks(self, intensity: str, length_ql: float) -> list[dict[str, Any]]:
        measure_len = self.bar_length
        pedal_value = 127 if intensity == "high" else 64
        events = []
        t = 0.0
        while t < length_ql:
            events.append({"time": t, "cc": 64, "val": pedal_value})
            events.append({"time": min(length_ql, t + measure_len), "cc": 64, "val": 0})
            t += measure_len
        return events

    def _post_process_generated_part(
        self, part: stream.Part, section: dict[str, Any], ratio: float | None
    ) -> None:
        hand = None
        if part.id and "rh" in part.id.lower():
            hand = "RH"
        elif part.id and "lh" in part.id.lower():
            hand = "LH"
        if not hand:
            return

        intensity = section.get("musical_intent", {}).get("intensity", "medium")
        notes = list(part.recurse().notes)
        scaled = self._density_engine.scale_density(notes, str(intensity))
        note_offsets = []
        for n in scaled:
            cp = copy.deepcopy(n)
            if hasattr(cp, "activeSite"):
                cp.activeSite = None
            note_offsets.append((float(n.offset), cp))
        for n in notes:
            part.remove(n)
        for off, n in note_offsets:
            part.insert(off, n)
