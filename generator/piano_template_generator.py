from __future__ import annotations

import copy
import re
import statistics
from typing import Any

from music21 import articulations, chord, expressions, harmony, note, spanner, stream, volume

from utilities import humanizer
from utilities.cc_tools import merge_cc_events, finalize_cc_events
from utilities.tone_shaper import ToneShaper
from utilities.humanizer import apply_swing
from utilities.pedalizer import generate_pedal_cc

from .base_part_generator import BasePartGenerator
from .voicing_density import VoicingDensityEngine
from .articulation import ArticulationEngine, QL_32ND

PPQ = 480

class PianoTemplateGenerator(BasePartGenerator):
    """Very simple piano generator for alpha testing."""

    def __init__(
        self,
        *args,
        enable_articulation: bool = True,
        tone_preset: str | None = None,
        normalize_loudness: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._density_engine = VoicingDensityEngine()
        self._art_engine = ArticulationEngine()
        self.enable_articulation = enable_articulation
        self.tone_preset = tone_preset
        self.normalize_loudness = normalize_loudness

    def compose(
        self,
        *,
        section_data: dict[str, Any],
        overrides_root: Any | None = None,
        groove_profile_path: str | None = None,
        next_section_data: dict[str, Any] | None = None,
        part_specific_humanize_params: dict[str, Any] | None = None,
        shared_tracks: dict[str, Any] | None = None,
    ) -> stream.Part | dict[str, stream.Part]:
        result = super().compose(
            section_data=section_data,
            overrides_root=overrides_root,
            groove_profile_path=groove_profile_path,
            next_section_data=next_section_data,
            part_specific_humanize_params=part_specific_humanize_params,
            shared_tracks=shared_tracks,
        )

        chord_label = section_data.get("chord_symbol_for_voicing", "")
        tags = str(chord_label).lower().split()

        if self.enable_articulation and isinstance(result, dict) and tags:
            q_len = float(section_data.get("q_length", self.bar_length))
            try:
                start_cs = harmony.ChordSymbol(str(chord_label).split()[0])
                start_root = start_cs.root() or harmony.ChordSymbol("C").root()
            except Exception:
                start_root = harmony.ChordSymbol("C").root()
            rh = result.get("piano_rh")

            if "gliss" in tags and next_section_data:
                end_label = next_section_data.get("chord_symbol_for_voicing", "C")
                try:
                    end_cs = harmony.ChordSymbol(str(end_label).split()[0])
                    end_root = end_cs.root() or start_root
                except Exception:
                    end_root = start_root
                notes = self._art_engine.generate_gliss(start_root, end_root, q_len)
                for n, t in notes:
                    rh.insert(t, n)

            if "trill" in tags:
                notes = self._art_engine.generate_trill(start_root, q_len)
                for n, t in notes:
                    rh.insert(t, n)
                if self._art_engine.cc_events:
                    for p in result.values():
                        base = getattr(p, "extra_cc", [])
                        p.extra_cc = base + [
                            {"time": tt, "cc": cc, "val": vv}
                            for tt, cc, vv in self._art_engine.cc_events
                        ]

        if isinstance(result, dict):
            for p in result.values():
                finalize_cc_events(p)
        else:
            finalize_cc_events(result)
        return result

    def _render_part(
        self, section_data: dict[str, Any], next_section_data: dict[str, Any] | None = None
    ) -> dict[str, stream.Part]:
        chord_label = section_data.get("chord_symbol_for_voicing", "C")
        tags = set(re.findall(r"<(gliss(?:_up|_down)?|trill)>", chord_label))
        chord_label = re.sub(r"<[^>]+>", "", chord_label)
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

        # Density scaling before pedal/humanizer logic
        for part in (rh, lh):
            notes = list(part.recurse().notes)
            scaled = self._density_engine.scale_density(notes, str(intensity))
            for n in notes:
                part.remove(n)
            for n in scaled:
                cp = copy.deepcopy(n)
                if hasattr(cp, "activeSite"):
                    cp.activeSite = None
                part.insert(float(n.offset), cp)

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

        if "trill" in tags:
            for part in (rh, lh):
                for n in part.recurse().notes:
                    n.articulations.append(articulations.Trill())
        for tag in tags:
            if tag.startswith("gliss"):
                direction = None
                if tag.endswith("_up"):
                    direction = "up"
                elif tag.endswith("_down"):
                    direction = "down"
                for part in (rh, lh):
                    notes = list(part.recurse().notes)
                    for a, b in zip(notes, notes[1:]):
                        if direction == "up" and b.pitch.midi <= a.pitch.midi:
                            continue
                        if direction == "down" and b.pitch.midi >= a.pitch.midi:
                            continue
                        part.insert(0, spanner.Glissando(a, b))

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
        from utilities.loudness_normalizer import normalize_velocities

        notes = list(part.recurse().notes)
        if notes and self.normalize_loudness:
            normalize_velocities(notes)
            import statistics

            intensity = section.get("musical_intent", {}).get("intensity", "medium")
            avg_vel = statistics.mean(n.volume.velocity or 64 for n in notes)
            shaper = ToneShaper()
            preset = self.tone_preset or shaper.choose_preset(
                intensity=intensity,
                avg_velocity=avg_vel,
            )
            existing = [
                (e["time"], e["cc"], e["val"]) if isinstance(e, dict) else e
                for e in getattr(part, "extra_cc", [])
                if (e.get("cc") if isinstance(e, dict) else e[1]) != 31
            ]
            tone_events = shaper.to_cc_events(
                amp_name=preset,
                intensity=intensity,
                as_dict=False,
            )
            part.extra_cc = merge_cc_events(set(existing), set(tone_events))
        return
