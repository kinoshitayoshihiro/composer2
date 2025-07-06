"""Simple block-chord generator for string ensembles."""

from __future__ import annotations

import copy
import enum
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import music21.articulations as articulations
import music21.expressions as expressions
import music21.spanner as m21spanner
from music21 import (
    chord,
    harmony,
    interval,
    note,
    pitch,
    stream,
    tie,
    volume,
)
from music21 import (
    instrument as m21instrument,
)

from utilities.core_music_utils import (
    get_key_signature_object,
    get_time_signature_object,
)
from utilities.velocity_curve import interpolate_7pt, resolve_velocity_curve
from utilities.cc_tools import finalize_cc_events

from .base_part_generator import BasePartGenerator


@dataclass(frozen=True)
class _SectionInfo:
    name: str
    instrument: m21instrument.Instrument
    range_low: str
    range_high: str
    velocity_pos: float


class BowPosition(enum.StrEnum):
    """Bow position enumeration."""

    TASTO = "tasto"
    NORMALE = "normale"
    PONTICELLO = "ponticello"


def parse_articulation_field(field: Any) -> list[str]:
    """Parse an articulation specification string into a list of names."""
    if field is None:
        return []
    if isinstance(field, (list | tuple | set)):
        result: list[str] = []
        for item in field:
            result.extend(parse_articulation_field(item))
        return result
    text = str(field)
    tokens = [t for t in re.split(r"[+\s]+", text) if t]
    return tokens


def parse_bow_position(value: Any) -> BowPosition | None:
    """Convert *value* into a :class:`BowPosition` or ``None``."""
    if value is None:
        return None
    try:
        return BowPosition(str(value).lower())
    except Exception:
        return BowPosition.NORMALE


class StringsGenerator(BasePartGenerator):
    """Generate very simple block-chord lines for a standard string section."""

    _SECTIONS = [
        _SectionInfo("contrabass", m21instrument.Contrabass(), "C1", "C3", 0.4),
        _SectionInfo("violoncello", m21instrument.Violoncello(), "C2", "E4", 0.4),
        _SectionInfo("viola", m21instrument.Viola(), "C3", "A5", 0.6),
        _SectionInfo("violin_ii", m21instrument.Violin(), "G3", "D7", 0.6),
        _SectionInfo("violin_i", m21instrument.Violin(), "G3", "D7", 0.8),
    ]

    def __init__(
        self,
        *,
        global_settings: dict | None = None,
        default_instrument: m21instrument.Instrument | None = None,
        global_tempo: int | None = None,
        global_time_signature: str | None = None,
        global_key_signature_tonic: str | None = None,
        global_key_signature_mode: str | None = None,
        voice_allocation: dict[str, int] | None = None,
        default_velocity_curve: list[int] | list[float] | str | None = None,
        voicing_mode: str = "close",
        divisi: bool | dict[str, str] | None = None,
        avoid_low_open_strings: bool = False,
        timing_jitter_ms: float = 0.0,
        timing_jitter_mode: str = "uniform",
        timing_jitter_scale_mode: str = "absolute",
        balance_scale: float = 1.0,
        rng=None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            global_settings=global_settings,
            default_instrument=default_instrument or m21instrument.Violin(),
            global_tempo=global_tempo,
            global_time_signature=global_time_signature,
            global_key_signature_tonic=global_key_signature_tonic,
            global_key_signature_mode=global_key_signature_mode,
            rng=rng,
            **kwargs,
        )
        ts_obj = get_time_signature_object(global_time_signature)
        self.measure_duration = (
            ts_obj.barDuration.quarterLength if ts_obj else self.bar_length
        )
        self.bar_length = self.measure_duration
        from collections.abc import Sequence

        self.voice_allocation = voice_allocation or {}
        if (
            isinstance(default_velocity_curve, Sequence)
            and not isinstance(default_velocity_curve, str)
        ):
            self.default_velocity_curve = self._prepare_velocity_map(
                default_velocity_curve
            )
        else:
            self.default_velocity_curve = self._select_velocity_curve(
                default_velocity_curve
            )
        self.voicing_mode = str(voicing_mode or "close").lower()
        self.divisi = divisi
        self.avoid_low_open_strings = bool(avoid_low_open_strings)
        self.timing_jitter_ms = float(timing_jitter_ms)
        self.timing_jitter_mode = str(timing_jitter_mode or "uniform").lower()
        self.timing_jitter_scale_mode = str(timing_jitter_scale_mode or "absolute").lower()
        self.balance_scale = float(balance_scale)
        self._last_parts: dict[str, stream.Part] | None = None
        self._articulation_map = {
            "sustain": None,
            "staccato": articulations.Staccato(),
            "accent": articulations.Accent(),
            "tenuto": articulations.Tenuto(),
            "legato": m21spanner.Slur(),
            "tremolo": expressions.Tremolo(),
            "pizz": expressions.TextExpression("pizz."),
            "arco": expressions.TextExpression("arco"),
        }
        self._legato_active: dict[str, list[note.NotRest]] = {}
        self._legato_groups: dict[str, list[tuple[note.NotRest, note.NotRest]]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def compose(
        self, *, section_data: dict[str, Any], **kwargs: Any
    ) -> dict[str, stream.Part]:
        result = super().compose(section_data=section_data, **kwargs)
        if not isinstance(result, dict):
            raise RuntimeError(
                "StringsGenerator expected dict result from _render_part"
            )
        for part in result.values():
            finalize_cc_events(part)
        self._last_parts = result
        return result

    def export_musicxml(self, path: str) -> None:
        if not self._last_parts:
            raise ValueError("No generated parts available for export")
        score = stream.Score()
        for info in self._SECTIONS:
            part = self._last_parts.get(info.name)
            if part is None:
                part = stream.Part(id=info.name)
                part.partName = f"Empty {info.name.replace('_', ' ').title()}"
                part.insert(0, info.instrument)
            score.insert(0, part)
        score.write("musicxml", fp=path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _prepare_velocity_map(self, spec: Any) -> list[int]:
        """Return a normalized 128-point velocity curve."""
        curve = resolve_velocity_curve(spec)
        if not curve:
            curve = self._default_log_curve()
        if all(0.0 <= v <= 1.5 for v in curve):
            curve = [int(127 * v) for v in curve]
        else:
            curve = [int(v) for v in curve]
        if len(curve) == 3:
            c0, c1, c2 = curve
            curve = [
                c0 + (c1 - c0) * (i / 3) if i <= 3 else c1 + (c2 - c1) * ((i - 3) / 3)
                for i in range(7)
            ]
        if len(curve) == 7:
            curve = interpolate_7pt(curve)
        if len(curve) != 128:
            result: list[int] = []
            for i in range(128):
                pos = i / 127 * (len(curve) - 1)
                idx0 = int(pos)
                idx1 = min(len(curve) - 1, idx0 + 1)
                frac = pos - idx0
                val = curve[idx0] * (1 - frac) + curve[idx1] * frac
                result.append(int(round(val)))
            curve = result
        return [max(0, min(127, int(v))) for v in curve]

    def _select_velocity_curve(self, style: Sequence[int | float] | str | None) -> list[int]:
        """Return velocity curve list for *style* or fallback."""
        from collections.abc import Sequence as _Seq

        if style is None:
            base = getattr(self, "default_velocity_curve", None)
            if base is not None:
                return list(base)
            return self._default_log_curve()

        if not isinstance(style, str) and isinstance(style, _Seq):
            try:
                import numpy as np  # type: ignore

                if isinstance(style, np.ndarray):
                    curve = style.tolist()
                else:
                    curve = list(style)
            except Exception:
                curve = list(style)
            return self._prepare_velocity_map(curve)

        curve = resolve_velocity_curve(style)
        if not curve:
            return self._default_log_curve()
        return self._prepare_velocity_map(curve)

    @staticmethod
    def _default_log_curve() -> list[int]:
        min_v, max_v = 32, 112
        result = []
        for i in range(128):
            frac = math.log1p(i) / math.log1p(127)
            result.append(int(round(min_v + (max_v - min_v) * frac)))
        return result

    @staticmethod
    def _fit_pitch(
        p: pitch.Pitch, low: int, high: int, above: int | None
    ) -> pitch.Pitch:
        base = p.midi
        n_min = math.ceil((low - base) / 12)
        n_max = math.floor((high - base) / 12)
        candidates = [base + 12 * n for n in range(n_min, n_max + 1)]
        if not candidates:
            val = max(min(base, high), low)
            return pitch.Pitch(midi=int(val))
        if above is not None:
            near = [c for c in candidates if abs(c - above) <= 4]
            if near:
                val = min(near, key=lambda x: abs(x - above))
            else:
                val = min(candidates, key=lambda x: abs(x - above))
        else:
            val = min(candidates, key=lambda x: abs(x - base))
        return pitch.Pitch(midi=int(val))

    def _estimate_velocity(self, section: _SectionInfo) -> tuple[int, float] | None:
        """Return (base velocity, factor) for *section*."""
        if not self.default_velocity_curve:
            return None
        idx = min(127, int(round(127 * section.velocity_pos)))
        base = self.default_velocity_curve[idx]
        factor = section.velocity_pos * self.balance_scale
        return base, factor

    def _velocity_for(self, section: _SectionInfo) -> int | None:
        result = self._estimate_velocity(section)
        if result is None:
            return None
        base, factor = result
        val = base * factor
        val = max(20, min(127, int(round(val))))
        return val

    def _humanize_timing(
        self,
        el: note.NotRest,
        jitter_ms: float,
        *,
        scale_mode: str = "absolute",
    ) -> None:
        if not jitter_ms:
            return
        jitter_val = jitter_ms
        reference_bpm = 120.0
        if scale_mode == "bpm_relative":
            current = float(self.global_tempo or reference_bpm)
            jitter_val *= reference_bpm / current
        if self.timing_jitter_mode == "gauss":
            jitter = self.rng.gauss(0.0, jitter_val / 2.0)
        else:
            jitter = self.rng.uniform(-jitter_val / 2.0, jitter_val / 2.0)
        ql_shift = jitter * reference_bpm / 60000.0
        new_offset = float(el.offset) + ql_shift
        el.offset = max(0.0, new_offset)

    # ------------------------------------------------------------------
    # Articulation helpers
    # ------------------------------------------------------------------
    def _handle_legato(
        self, part_name: str, note_obj: note.NotRest, apply: bool
    ) -> None:
        if apply:
            buf = self._legato_active.get(part_name)
            if buf:
                buf[1] = note_obj
            else:
                self._legato_active[part_name] = [note_obj, note_obj]
        else:
            buf = self._legato_active.pop(part_name, None)
            if buf and buf[0] != buf[1]:
                self._legato_groups.setdefault(part_name, []).append((buf[0], buf[1]))

    def _apply_articulations(
        self,
        elem: note.NotRest,
        art_names: Any,
        part_name: str,
    ) -> bool:
        names = parse_articulation_field(art_names)
        legato = False
        for art_name in names:
            art_obj = self._articulation_map.get(art_name)
            if art_obj is None:
                if art_name in self._articulation_map:
                    continue
                self.logger.warning("Unknown articulation '%s'", art_name)
                continue
            if art_name == "legato":
                legato = True
            elif art_name in {"pizz", "arco"}:
                elem.expressions.append(copy.deepcopy(art_obj))
            elif art_name == "tremolo" and isinstance(elem, chord.Chord):
                trem = copy.deepcopy(art_obj)
                if hasattr(trem, "rapid"):
                    trem.rapid = True
                elem.expressions.append(trem)
            else:
                elem.articulations.append(copy.deepcopy(art_obj))
        self._handle_legato(part_name, elem, legato)
        if not legato:
            self._handle_legato(part_name, elem, False)
        return legato

    def _create_notes_from_event(
        self,
        base_pitch: pitch.Pitch | chord.Chord,
        duration_ql: float,
        part_name: str,
        event_articulations: list[str] | None,
        velocity: int | None,
        velocity_factor: float = 1.0,
        bow_position: BowPosition | None = None,
    ) -> note.NotRest:
        if isinstance(base_pitch, chord.Chord):
            n: note.NotRest = chord.Chord(base_pitch.pitches, quarterLength=duration_ql)
        else:
            n = note.Note(base_pitch, quarterLength=duration_ql)
        if velocity is not None:
            final_vel = max(1, min(127, int(round(velocity * velocity_factor))))
            vol = volume.Volume(velocity=final_vel)
            try:
                vol.velocityScalar = final_vel / 127.0
            except Exception:
                pass
            if hasattr(vol, "expressiveDynamic"):
                try:
                    vol.expressiveDynamic = final_vel / 127.0
                except Exception:
                    pass
            n.volume = vol
        if bow_position:
            value = bow_position.value
            if hasattr(n.style, "bowPosition"):
                setattr(n.style, "bowPosition", value)
            else:
                setattr(n.style, "other", value)

        return n

    def _finalize_part(self, part: stream.Part, part_name: str) -> stream.Part:
        buf = self._legato_active.pop(part_name, None)
        if buf and buf[0] != buf[1]:
            self._legato_groups.setdefault(part_name, []).append((buf[0], buf[1]))
        for start, end in self._legato_groups.get(part_name, []):
            try:
                part.insert(0, m21spanner.Slur(start, end))
            except Exception:
                pass
        self._legato_groups[part_name] = []
        return part

    def _split_durations(self, q_len: float) -> list[float]:
        remaining = q_len
        segments: list[float] = []
        while remaining > 0:
            if remaining > self.bar_length:
                seg = self.bar_length * 0.95
            else:
                seg = remaining
            segments.append(seg)
            remaining -= seg
        if not segments:
            segments.append(q_len)
        total = sum(segments)
        if abs(total - q_len) > 1e-6:
            segments[-1] += q_len - total
        return segments

    def _voiced_pitches(self, cs: harmony.ChordSymbol) -> list[pitch.Pitch]:
        pitches_sorted = sorted(
            {p.pitchClass: p for p in cs.pitches}.values(), key=lambda p: p.midi
        )
        if self.voicing_mode == "open":
            voiced = [p.transpose(12 * (i // 2)) for i, p in enumerate(pitches_sorted)]
        elif self.voicing_mode == "spread":
            voiced = [p.transpose(12 * i) for i, p in enumerate(pitches_sorted)]
        else:
            voiced = pitches_sorted
        return voiced

    # ------------------------------------------------------------------
    # Core rendering
    # ------------------------------------------------------------------
    def _render_part(
        self,
        section_data: dict[str, Any],
        next_section_data: dict[str, Any] | None = None,
    ) -> dict[str, stream.Part]:
        chord_label = (
            section_data.get("chord_symbol_for_voicing")
            or section_data.get("original_chord_label")
            or "C"
        )
        q_length = float(section_data.get("q_length", self.bar_length))
        events = section_data.get("events")
        if events:
            durations = [float(e.get("duration", 1.0)) for e in events]
            event_articulations = [e.get("articulations") for e in events]
        else:
            durations = self._split_durations(q_length)
            event_articulations = [None] * len(durations)
        try:
            cs = harmony.ChordSymbol(chord_label)
            base_pitches = self._voiced_pitches(cs)
        except Exception as exc:  # pragma: no cover - parsing errors are logged
            self.logger.error("Invalid chord '%s': %s", chord_label, exc)
            base_pitches = []

        parts: dict[str, stream.Part] = {}
        if not base_pitches:
            for info in self._SECTIONS:
                part = stream.Part(id=info.name)
                part.insert(0, info.instrument)
                for dur in durations:
                    part.append(note.Rest(quarterLength=dur))
                parts[info.name] = part
            return parts

        while len(base_pitches) < len(self._SECTIONS):
            base_pitches.append(base_pitches[len(base_pitches) % len(base_pitches)])

        extras_map: dict[str, list[pitch.Pitch]] = {s.name: [] for s in self._SECTIONS}
        if not self.divisi and len(base_pitches) > len(self._SECTIONS):
            extras = base_pitches[len(self._SECTIONS):]
            target_sections = ["violin_i", "violin_ii", "viola"]
            t_idx = 0
            for p_extra in extras:
                for _ in range(len(target_sections)):
                    sec_name = target_sections[t_idx % len(target_sections)]
                    sec_info = next(s for s in self._SECTIONS if s.name == sec_name)
                    low = pitch.Pitch(sec_info.range_low).midi
                    high = pitch.Pitch(sec_info.range_high).midi
                    adj_extra = self._fit_pitch(p_extra, low, high, None)
                    if low <= adj_extra.midi <= high:
                        extras_map[sec_name].append(adj_extra)
                        t_idx += 1
                        break
                    t_idx += 1

        prev_midi: int | None = None
        divisi_map: dict[str, str] = {}
        if isinstance(self.divisi, bool) and self.divisi:
            divisi_map = {"violin_i": "octave", "violin_ii": "octave"}
        elif isinstance(self.divisi, dict):
            divisi_map = {k: str(v) for k, v in self.divisi.items()}

        default_arts = (
            section_data.get("part_params", {})
            .get("strings", {})
            .get("default_articulations")
        )

        for idx, info in enumerate(self._SECTIONS):
            pitch_idx = self.voice_allocation.get(info.name, idx)
            part = stream.Part(id=info.name)
            part.partName = info.name.replace("_", " ").title()
            part.insert(0, info.instrument)

            if pitch_idx is None or pitch_idx < 0:
                for dur in durations:
                    part.append(note.Rest(quarterLength=dur))
                parts[info.name] = part
                continue

            src = base_pitches[pitch_idx % len(base_pitches)]
            low = pitch.Pitch(info.range_low).midi
            high = pitch.Pitch(info.range_high).midi
            adj = self._fit_pitch(src, low, high, prev_midi)
            if self.avoid_low_open_strings and info.name in {
                "violin_i",
                "violin_ii",
                "viola",
            }:
                name_oct = adj.nameWithOctave
                if info.name == "viola" and name_oct in {"C3", "G3"}:
                    if adj.midi + 12 <= high:
                        adj = adj.transpose(12)
                elif info.name != "viola" and name_oct == "G3":
                    if adj.midi + 12 <= high:
                        adj = adj.transpose(12)
            vel_info = self._estimate_velocity(info)
            if vel_info is None:
                vel_base, vel_factor = None, 1.0
            else:
                vel_base, vel_factor = vel_info
            offset = 0.0
            prev_note: note.NotRest | None = None
            for i, dur in enumerate(durations):
                arts = event_articulations[i] if i < len(event_articulations) else None
                if not arts:
                    arts = default_arts
                bow_pos = None
                if events and i < len(events):
                    bow_pos = parse_bow_position(events[i].get("bow_position"))
                if bow_pos is None:
                    bow_pos = parse_bow_position(section_data.get("bow_position"))
                base_obj: pitch.Pitch | chord.Chord
                pitch_list = [adj]
                if extras_map.get(info.name):
                    pitch_list.extend(extras_map[info.name])
                base_obj = chord.Chord(pitch_list) if len(pitch_list) > 1 else adj
                n = self._create_notes_from_event(
                    base_obj,
                    dur,
                    info.name,
                    arts,
                    vel_base,
                    vel_factor,
                    bow_pos,
                )
                is_legato = self._apply_articulations(n, arts, info.name)
                self._humanize_timing(
                    n,
                    self.timing_jitter_ms,
                    scale_mode=self.timing_jitter_scale_mode,
                )
                if len(durations) > 1:
                    if i == 0:
                        n.tie = tie.Tie("start")
                    elif i == len(durations) - 1:
                        n.tie = tie.Tie("stop")
                    else:
                        n.tie = tie.Tie("continue")
                elem: note.NotRest = n
                if info.name in divisi_map:
                    mode = divisi_map[info.name]
                    if mode == "octave":
                        extra_pitch = n.pitch.transpose(12)
                    elif mode == "third":
                        key_obj = get_key_signature_object(
                            self.global_key_signature_tonic,
                            self.global_key_signature_mode,
                        )
                        if key_obj:
                            deg = key_obj.getScaleDegreeFromPitch(n.pitch)
                            if deg is not None:
                                extra_pitch = key_obj.pitchFromDegree(deg + 2)
                                extra_pitch.octave = n.pitch.octave
                                if extra_pitch.midi <= n.pitch.midi:
                                    extra_pitch.octave += 1
                            else:
                                qual = "M3" if key_obj.mode == "major" else "m3"
                                extra_pitch = interval.Interval(qual).transposePitch(
                                    n.pitch
                                )
                        else:
                            qual = (
                                "M3"
                                if (self.global_key_signature_mode or "major")
                                == "major"
                                else "m3"
                            )
                            extra_pitch = interval.Interval(qual).transposePitch(
                                n.pitch
                            )
                    else:
                        extra_pitch = n.pitch.transpose(4)
                    if extra_pitch:
                        if extra_pitch.midi > high:
                            if extra_pitch.midi - 12 >= low:
                                extra_pitch = extra_pitch.transpose(-12)
                            else:
                                extra_pitch = None
                    if extra_pitch:
                        chd = chord.Chord([n.pitch, extra_pitch])
                        chd.quarterLength = n.quarterLength
                        chd.offset = n.offset
                        if n.tie:
                            chd.tie = n.tie
                        if n.volume and n.volume.velocity is not None:
                            chd.volume = volume.Volume(velocity=int(n.volume.velocity))
                        elem = chd
                if not is_legato:
                    if (
                        prev_note
                        and not prev_note.isRest
                        and not n.isRest
                        and prev_note.quarterLength >= 0.5
                        and n.quarterLength >= 0.5
                        and abs(n.pitch.midi - prev_note.pitch.midi) <= 2
                    ):
                        self._handle_legato(info.name, prev_note, True)
                        self._handle_legato(info.name, n, True)
                        self._handle_legato(info.name, n, False)
                    else:
                        self._handle_legato(info.name, n, False)
                part.insert(offset + float(n.offset), elem)
                offset += dur
                prev_note = n if not n.isRest else None
            parts[info.name] = self._finalize_part(part, info.name)
            prev_midi = adj.midi
        dim_start = section_data.get("dim_start")
        dim_end = section_data.get("dim_end")
        crescendo_flag = section_data.get("crescendo", q_length >= self.bar_length)
        if dim_start is not None and dim_end is not None:
            self._apply_expression_cc(
                parts,
                crescendo=bool(crescendo_flag),
                length=q_length,
                start_val=int(dim_start),
                end_val=int(dim_end),
            )
        elif q_length >= 0:
            self._apply_expression_cc(parts, crescendo=bool(crescendo_flag), length=q_length)
        return parts

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _merge_identical_bars(self, part: stream.Part) -> stream.Part:
        """Merge consecutive bars with identical content."""
        meas = part.makeMeasures(inPlace=False)
        measures = list(meas.getElementsByClass(stream.Measure))
        if len(measures) <= 1:
            return part

        def _sig(m: stream.Measure) -> tuple:
            items = []
            for el in m.notesAndRests:
                if isinstance(el, chord.Chord):
                    name = tuple(p.nameWithOctave for p in el.pitches)
                elif isinstance(el, note.Note):
                    name = el.pitch.nameWithOctave
                else:
                    name = "Rest"
                items.append(
                    (
                        float(el.offset),
                        name,
                        float(el.quarterLength),
                        el.tie.type if el.tie else None,
                    )
                )
            return tuple(items)

        new_measures: list[stream.Measure] = []
        prev_sig: tuple | None = None
        for m in measures:
            sig = _sig(m)
            if new_measures and sig == prev_sig:
                last = new_measures[-1]
                for e_prev, e_cur in zip(last.notesAndRests, m.notesAndRests):
                    e_prev.quarterLength += e_cur.quarterLength
                    if e_prev.tie:
                        if e_cur.tie and e_cur.tie.type == "stop":
                            e_prev.tie.type = "stop"
                        else:
                            if e_prev.tie.type == "start":
                                e_prev.tie.type = "start"
                    elif e_cur.tie:
                        e_prev.tie = tie.Tie(e_cur.tie.type)
            else:
                new_measures.append(m)
                prev_sig = sig

        new_part = stream.Part(id=part.id)
        for m in part.recurse().getElementsByClass(m21instrument.Instrument):
            new_part.insert(0, m)
            break
        offset = 0.0
        import copy

        for m in new_measures:
            for el in m.notesAndRests:
                new_el = copy.deepcopy(el)
                new_part.insert(offset + el.offset, new_el)
            offset += m.duration.quarterLength
        return new_part

    def _apply_expression_cc(
        self,
        parts: dict[str, stream.Part],
        crescendo: bool = True,
        *,
        length: float | None = None,
        start_val: int | None = None,
        end_val: int | None = None,
    ) -> None:
        """Add a CC11 envelope across *length* quarter lengths.

        Parameters
        ----------
        parts:
            Mapping of part names to ``music21`` Parts.
        crescendo:
            If ``True`` and no explicit ``start_val``/``end_val`` provided,
            ramp from 64 to 80, else from 80 to 64.
        length:
            Envelope duration in quarter lengths. Defaults to bar length.
        start_val:
            Starting CC11 value. Overrides ``crescendo`` when provided.
        end_val:
            Ending CC11 value. Overrides ``crescendo`` when provided.
        """
        from utilities.cc_tools import merge_cc_events

        if length is None:
            length = self.bar_length
        if start_val is None or end_val is None:
            start_val = 64 if crescendo else 80
            end_val = 80 if crescendo else 64

        for p in parts.values():
            events = [
                (0.0, 11, int(start_val)),
                (float(length), 11, int(end_val)),
            ]
            base_events = getattr(p, "_extra_cc", set())
            merged = merge_cc_events(base_events, events)
            p._extra_cc = set(merged)
