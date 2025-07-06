"""Simple block-chord generator for string ensembles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict
import math

from music21 import harmony, instrument as m21instrument, note, pitch, stream, volume, chord, tie, interval

from utilities.velocity_curve import interpolate_7pt, resolve_velocity_curve

from .base_part_generator import BasePartGenerator
from utilities.core_music_utils import get_time_signature_object, get_key_signature_object


@dataclass(frozen=True)
class _SectionInfo:
    name: str
    instrument: m21instrument.Instrument
    range_low: str
    range_high: str
    velocity_pos: float


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
        voice_allocation: Dict[str, int] | None = None,
        default_velocity_curve: list[int] | list[float] | str | None = None,
        voicing_mode: str = "close",
        divisi: bool | Dict[str, str] | None = None,
        avoid_low_open_strings: bool = False,
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
        self.measure_duration = ts_obj.barDuration.quarterLength if ts_obj else self.bar_length
        self.bar_length = self.measure_duration
        self.voice_allocation = voice_allocation or {}
        self.default_velocity_curve = self._prepare_velocity_map(default_velocity_curve)
        self.voicing_mode = str(voicing_mode or "close").lower()
        self.divisi = divisi
        self.avoid_low_open_strings = bool(avoid_low_open_strings)
        self._last_parts: Dict[str, stream.Part] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def compose(self, *, section_data: Dict[str, Any], **kwargs: Any) -> Dict[str, stream.Part]:
        result = super().compose(section_data=section_data, **kwargs)
        if not isinstance(result, dict):
            raise RuntimeError("StringsGenerator expected dict result from _render_part")
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
    def _prepare_velocity_map(self, spec: Any) -> list[int] | None:
        curve = resolve_velocity_curve(spec)
        if not curve:
            if spec is None:
                curve = self._default_log_curve()
            else:
                return None
        if all(0.0 <= v <= 1.5 for v in curve):
            curve = [int(127 * v) for v in curve]
        else:
            curve = [int(v) for v in curve]
        if len(curve) == 3:
            c0, c1, c2 = curve
            curve = [c0 + (c1 - c0) * (i / 3) if i <= 3 else c1 + (c2 - c1) * ((i - 3) / 3) for i in range(7)]
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

    @staticmethod
    def _default_log_curve() -> list[int]:
        min_v, max_v = 32, 112
        result = []
        for i in range(128):
            frac = math.log1p(i) / math.log1p(127)
            result.append(int(round(min_v + (max_v - min_v) * frac)))
        return result

    @staticmethod
    def _fit_pitch(p: pitch.Pitch, low: int, high: int, above: int | None) -> pitch.Pitch:
        q = p.transpose(0)
        while q.midi < low:
            q = q.transpose(12)
        while q.midi > high:
            q = q.transpose(-12)
        if above is not None:
            while q.midi <= above and q.midi + 12 <= high:
                q = q.transpose(12)
        if q.midi > high:
            q = pitch.Pitch(midi=high)
        return q

    def _velocity_for(self, section: _SectionInfo) -> int | None:
        if not self.default_velocity_curve:
            return None
        idx = int(round(127 * section.velocity_pos))
        idx = min(127, idx)
        base = self.default_velocity_curve[idx]
        val = base + (127 - base) * (section.velocity_pos - 0.5) * 0.25
        val = max(1, min(127, val))
        return int(round(val))

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
        pitches_sorted = sorted({p.pitchClass: p for p in cs.pitches}.values(), key=lambda p: p.midi)
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
        self, section_data: Dict[str, Any], next_section_data: Dict[str, Any] | None = None
    ) -> Dict[str, stream.Part]:
        chord_label = (
            section_data.get("chord_symbol_for_voicing")
            or section_data.get("original_chord_label")
            or "C"
        )
        q_length = float(section_data.get("q_length", self.bar_length))
        durations = self._split_durations(q_length)
        try:
            cs = harmony.ChordSymbol(chord_label)
            base_pitches = self._voiced_pitches(cs)
        except Exception as exc:  # pragma: no cover - parsing errors are logged
            self.logger.error("Invalid chord '%s': %s", chord_label, exc)
            base_pitches = []

        parts: Dict[str, stream.Part] = {}
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

        prev_midi: int | None = None
        divisi_map: Dict[str, str] = {}
        if isinstance(self.divisi, bool) and self.divisi:
            divisi_map = {"violin_i": "octave", "violin_ii": "octave"}
        elif isinstance(self.divisi, dict):
            divisi_map = {k: str(v) for k, v in self.divisi.items()}

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
            if self.avoid_low_open_strings and info.name in {"violin_i", "violin_ii", "viola"}:
                name_oct = adj.nameWithOctave
                if info.name == "viola" and name_oct in {"C3", "G3"}:
                    if adj.midi + 12 <= high:
                        adj = adj.transpose(12)
                elif info.name != "viola" and name_oct == "G3":
                    if adj.midi + 12 <= high:
                        adj = adj.transpose(12)
            vel = self._velocity_for(info)
            for i, dur in enumerate(durations):
                n = note.Note(adj)
                n.quarterLength = dur
                if len(durations) > 1:
                    if i == 0:
                        n.tie = tie.Tie("start")
                    elif i == len(durations) - 1:
                        n.tie = tie.Tie("stop")
                    else:
                        n.tie = tie.Tie("continue")
                if vel is not None:
                    n.volume = volume.Volume(velocity=vel)
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
                                extra_pitch = interval.Interval(qual).transposePitch(n.pitch)
                        else:
                            qual = "M3" if (self.global_key_signature_mode or "major") == "major" else "m3"
                            extra_pitch = interval.Interval(qual).transposePitch(n.pitch)
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
                        if n.tie:
                            chd.tie = n.tie
                        if vel is not None:
                            chd.volume = volume.Volume(velocity=vel)
                        elem = chd
                part.append(elem)
            parts[info.name] = part
            prev_midi = adj.midi
        if q_length >= self.bar_length:
            self._apply_expression_cc(parts, crescendo=True)
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
                    (float(el.offset), name, float(el.quarterLength), el.tie.type if el.tie else None)
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

    def _apply_expression_cc(self, parts: Dict[str, stream.Part], crescendo: bool = True) -> None:
        """Add a simple CC11 crescendo over one bar."""
        from utilities.cc_tools import merge_cc_events

        for p in parts.values():
            events = [
                (0.0, 11, 64 if crescendo else 80),
                (self.bar_length, 11, 80 if crescendo else 64),
            ]
            base: set[tuple[float, int, int]] = set(getattr(p, "_extra_cc", set()))
            p._extra_cc = set(merge_cc_events(base, events))

