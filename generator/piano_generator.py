# generator/piano_generator.py (v2.3 最終修正版)

from __future__ import annotations

import copy
import logging
import random
from typing import Any

from music21 import chord as m21chord
from music21 import duration as m21duration
from music21 import expressions, harmony, interval, note, pitch, stream
from music21 import volume as m21volume

from utilities.rest_utils import get_rest_windows
from utilities.velocity_curve import resolve_velocity_curve

# Pedalクラスはexpressionsから個別にインポート
try:
    from music21.expressions import PedalMark
except Exception:  # pragma: no cover - compatibility
    from music21.expressions import TextExpression as PedalMark

from utilities import humanizer

from .base_part_generator import BasePartGenerator

try:
    from utilities.core_music_utils import (
        MIN_NOTE_DURATION_QL,
        get_time_signature_object,
    )
    from utilities.override_loader import PartOverride
    from utilities.scale_registry import ScaleRegistry
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Required dependencies are missing. Please run 'pip install -r requirements.txt'."
    ) from e


logger = logging.getLogger("modular_composer.piano_generator")

# (LUTは変更なし)
EMO_TO_BUCKET_PIANO = {
    "quiet_pain": "calm",
    "nascent_strength": "calm",
    "hope_dawn": "calm",
    "emotional_realization": "groove",
    "gratitude": "groove",
    "wavering_heart": "groove",
    "love_and_resolution": "energetic",
    "trial_prayer": "energetic",
    "emotional_storm": "energetic",
    "default": "calm",
}
BUCKET_TO_PATTERN_PIANO = {
    ("calm", "low"): ("piano_rh_ambient_pad", "piano_lh_roots_whole"),
    ("calm", "medium"): ("piano_rh_block_chords_quarters", "piano_lh_roots_half"),
    ("calm", "medium_low"): ("piano_rh_block_chords_quarters", "piano_lh_roots_half"),
    ("calm", "high"): (
        "piano_rh_arpeggio_sixteenths_up_down",
        "piano_lh_octaves_quarters",
    ),
    ("groove", "low"): ("piano_rh_syncopated_chords_pop", "piano_lh_roots_half"),
    ("groove", "medium"): (
        "piano_rh_syncopated_chords_pop",
        "piano_lh_octaves_quarters",
    ),
    ("groove", "high"): (
        "piano_rh_arpeggio_sixteenths_up_down",
        "piano_lh_alberti_bass_eighths",
    ),
    ("energetic", "low"): (
        "piano_rh_block_chords_quarters",
        "piano_lh_octaves_quarters",
    ),
    ("energetic", "medium"): (
        "piano_rh_arpeggio_sixteenths_up_down",
        "piano_lh_alberti_bass_eighths",
    ),
    ("energetic", "high"): (
        "piano_rh_arpeggio_sixteenths_up_down",
        "piano_lh_octaves_quarters",
    ),
    ("default", "medium_low"): (
        "piano_rh_block_chords_quarters",
        "piano_lh_roots_half",
    ),
    ("default", "medium_high"): (
        "piano_rh_syncopated_chords_pop",
        "piano_lh_octaves_quarters",
    ),
    ("default", "high_to_very_high_then_fade"): (
        "piano_rh_arpeggio_sixteenths_up_down",
        "piano_lh_octaves_quarters",
    ),
    ("default", "default"): ("piano_rh_block_chords_quarters", "piano_lh_roots_whole"),
}


class PianoGenerator(BasePartGenerator):
    def __init__(
        self, *args, main_cfg=None, ml_velocity_model_path: str | None = None, **kwargs
    ):
        self.main_cfg = main_cfg
        self.part_parameters = kwargs.get("part_parameters", {})
        self.chord_voicer = None
        ts_obj = get_time_signature_object(kwargs.get("global_time_signature", "4/4"))
        self.global_time_signature_obj = ts_obj
        cfg_piano = (main_cfg or {}).get("piano", {})
        self.anticipatory_chord = bool(cfg_piano.get("anticipatory_chord", False))
        self.measure_duration = ts_obj.barDuration.quarterLength if ts_obj else 4.0
        super().__init__(*args, ml_velocity_model_path=ml_velocity_model_path, **kwargs)
        self.cfg: dict = kwargs.copy()
        self._prev_voicings = {"RH": None, "LH": None}
        self._prev_last_pitch = {"RH": None, "LH": None}
        self._style_cycle_index = {"RH": 0, "LH": 0}
        self._current_cycle_section = None
        # ...他の初期化処理...

    def _find_pattern_by_tags(self, tags: list[str], hand: str) -> str | None:
        for key, data in self.part_parameters.items():
            pat_tags = data.get("tags") or []
            if hand.lower() in pat_tags and all(t in pat_tags for t in tags):
                return key
        return None

    def _get_pattern_keys(
        self, musical_intent: dict[str, Any], overrides: PartOverride | None
    ) -> tuple[str, str]:
        emotion = musical_intent.get("emotion", "default")
        intensity = musical_intent.get("intensity", "medium")
        bucket = EMO_TO_BUCKET_PIANO.get(emotion, "default")
        keys = BUCKET_TO_PATTERN_PIANO.get((bucket, intensity))
        if not keys:
            base_intensity = str(intensity).split("_")[0]
            keys = BUCKET_TO_PATTERN_PIANO.get((bucket, base_intensity))
            if not keys:
                keys = BUCKET_TO_PATTERN_PIANO.get(("default", "default"))
        return keys

    def _render_hand_part(
        self,
        hand: str,
        cs: harmony.ChordSymbol,
        duration_ql: float,
        rhythm_key: str,
        params: dict[str, Any],
        voicing_style: str | None = None,
        mode: str | None = None,
        velocity_shift: int | None = None,
    ) -> stream.Part:
        part = stream.Part(id=f"Piano_{hand}")
        pattern_data = self.part_parameters.get(rhythm_key) or {}
        pattern_events = pattern_data.get("pattern") or []
        if not pattern_events:
            logger.warning(
                f"PianoGen: pattern '{rhythm_key}' not found or empty. Using "
                f"single root note fallback."
            )
            pattern_events = [
                {
                    "offset": 0.0,
                    "duration": duration_ql,
                    "type": "root",
                    "velocity_factor": 0.7,
                }
            ]

        options = pattern_data.get("options", {})
        velocity_curve_list = resolve_velocity_curve(options.get("velocity_curve"))

        octave = params.get(
            f"default_{hand.lower()}_target_octave", 4 if hand == "RH" else 2
        )
        if params.get("random_octave_shift"):
            octave += self.rng.choice([-1, 1])
        num_voices = params.get(
            f"default_{hand.lower()}_num_voices", 4 if hand == "RH" else 2
        )
        default_style = "spread" if hand == "RH" else "closed"
        chosen_style = voicing_style or params.get(
            f"default_{hand.lower()}_voicing_style",
            default_style,
        )
        voiced_pitches = self._voice_minimal_leap(
            hand, cs, num_voices, octave, chosen_style
        )

        scale_pitches = None
        if mode and cs.root():
            try:
                scl = ScaleRegistry.get(cs.root().name, mode)
                root_pitch = cs.root().transpose(0)
                root_pitch.octave = octave
                int_third = interval.Interval(
                    scl.pitchFromDegree(1), scl.pitchFromDegree(3)
                )
                int_fifth = interval.Interval(
                    scl.pitchFromDegree(1), scl.pitchFromDegree(5)
                )
                scale_pitches = {
                    "root": root_pitch,
                    "third": root_pitch.transpose(int_third),
                    "fifth": root_pitch.transpose(int_fifth),
                }
                logger.debug(f"Scale pitches for {hand}: {scale_pitches}")
            except Exception as e:
                logger.debug(f"scale pitch calc failed: {e}")
                scale_pitches = None

        if not voiced_pitches:
            part.insert(0, note.Rest(quarterLength=duration_ql))
            return part

        base_velocity = params.get("velocity", 70)
        ref_duration = float(pattern_data.get("length_beats", self.measure_duration))
        scale_factor = duration_ql / ref_duration if ref_duration > 0 else 1.0

        for p_event in pattern_events:
            offset = float(p_event.get("offset", 0.0)) * scale_factor
            duration = float(p_event.get("duration", 1.0)) * scale_factor
            if offset >= duration_ql:
                continue
            duration = min(duration, duration_ql - offset)
            if duration < MIN_NOTE_DURATION_QL / 4:
                continue
            vel_factor = float(p_event.get("velocity_factor", 1.0))
            velocity = int(base_velocity * vel_factor)
            if velocity_shift is not None:
                velocity += int(velocity_shift)
            layer_idx = p_event.get("velocity_layer")
            if velocity_curve_list and layer_idx is not None:
                try:
                    idx = int(layer_idx)
                    if 0 <= idx < len(velocity_curve_list):
                        velocity = int(velocity * velocity_curve_list[idx])
                except (TypeError, ValueError):
                    pass
            velocity = max(1, min(127, velocity))
            event_type = p_event.get("type", "chord")
            element_to_add = self._create_music_element(
                event_type, voiced_pitches, duration, hand, scale_pitches
            )
            if element_to_add:
                element_to_add.volume = m21volume.Volume(velocity=velocity)
                part.insert(offset, element_to_add)
        return part

    def _get_voiced_pitches(self, cs, num_voices, octave, style) -> list[pitch.Pitch]:
        if self.chord_voicer:
            try:
                return self.chord_voicer.voice_chord(
                    cs, style=style, num_voices=num_voices, target_octave=octave
                )
            except Exception as e:
                logger.warning(
                    f"ChordVoicer failed: {e}. Falling back to basic voicing."
                )

        cs_copy = copy.deepcopy(cs)
        cs_copy.closedPosition(inPlace=True)
        if not cs_copy.pitches:
            return []
        pitches = list(cs_copy.pitches)
        if style == "spread" and len(pitches) > 1:
            pitches = [pitches[0]] + [p.transpose(12) for p in pitches[1:]]
        elif style == "inverted" and len(pitches) > 0:
            pitches = pitches[1:] + [pitches[0].transpose(12)]

        bottom_pitch = pitches[0]
        target_pitch_ref = note.Note(bottom_pitch.name, octave=octave)
        transpose_interval = interval.Interval(bottom_pitch, target_pitch_ref)
        pitches = [p.transpose(transpose_interval) for p in pitches]

        return (
            pitches[:num_voices]
            if num_voices is not None and len(pitches) > num_voices
            else pitches
        )

    def _voice_minimal_leap(
        self,
        hand: str,
        cs: harmony.ChordSymbol,
        num_voices: int,
        octave: int,
        style: str,
    ) -> list[pitch.Pitch]:
        base_voicing = self._get_voiced_pitches(cs, num_voices, octave, style)
        if not base_voicing:
            return []
        prev = self._prev_voicings.get(hand)
        if prev is None:
            self._prev_voicings[hand] = base_voicing
            return base_voicing

        candidates: list[list[pitch.Pitch]] = []
        n = len(base_voicing)
        # generate inversions
        for inv in range(n):
            inv_pitches = []
            for i, p in enumerate(base_voicing):
                inv_pitches.append(p.transpose(12) if i < inv else p)
            for shift in (-12, 0, 12):
                candidates.append([pp.transpose(shift) for pp in inv_pitches])

        def distance(cand: list[pitch.Pitch]) -> float:
            cand_sorted = sorted(cand, key=lambda p: p.ps)
            prev_sorted = sorted(prev, key=lambda p: p.ps)
            m = min(len(cand_sorted), len(prev_sorted))
            return sum(abs(cand_sorted[i].ps - prev_sorted[i].ps) for i in range(m))

        best = min(candidates, key=distance)
        self._prev_voicings[hand] = best
        return best

    def _voice_leading(
        self, chords: list[harmony.ChordSymbol], mode: str = "shell"
    ) -> list[list[pitch.Pitch]]:
        """Return voice-led pitches for a chord progression."""
        prev: list[pitch.Pitch] | None = None
        result: list[list[pitch.Pitch]] = []
        for cs in chords:
            base: list[pitch.Pitch] = []
            if mode != "guide":
                root = cs.root() or harmony.ChordSymbol("C").root()
                if root:
                    rp = root.transpose(0)
                    rp.octave = rp.octave or 4
                    base.append(rp)
            if cs.third:
                tp = cs.third.transpose(0)
                tp.octave = tp.octave or 4
                base.append(tp)
            if cs.seventh:
                sp = cs.seventh.transpose(0)
                sp.octave = sp.octave or 4
                base.append(sp)
            elif cs.fifth:
                fp = cs.fifth.transpose(0)
                fp.octave = fp.octave or 4
                base.append(fp)

            if prev is None:
                voiced = base
            else:
                voiced = []
                for idx, p in enumerate(base):
                    target = prev[min(idx, len(prev) - 1)]
                    candidates = [p.transpose(o) for o in (-12, 0, 12)]
                    best = min(candidates, key=lambda c: abs(c.ps - target.ps))
                    voiced.append(best)
            prev = voiced
            result.append(voiced)
        return result

    def _create_music_element(
        self,
        event_type: str,
        pitches: list[pitch.Pitch],
        duration_ql: float,
        hand: str,
        scale_pitches: dict[str, pitch.Pitch] | None = None,
    ) -> stream.GeneralNote | None:
        if not pitches:
            return None

        def choose_pitch() -> pitch.Pitch:
            last = self._prev_last_pitch.get(hand)
            if last is None:
                return min(pitches, key=lambda p: p.ps)
            candidates = [p.transpose(oct) for p in pitches for oct in (-12, 0, 12)]
            return min(candidates, key=lambda p: abs(p.ps - last.ps))

        selected_pitch = choose_pitch()

        if event_type in ("octave_root", "octave"):
            elem = m21chord.Chord(
                [selected_pitch, selected_pitch.transpose(12)],
                quarterLength=duration_ql,
            )
        elif event_type == "root":
            elem = note.Note(selected_pitch, quarterLength=duration_ql)
        elif event_type == "root_octave_down" and scale_pitches:
            elem = note.Note(
                scale_pitches["root"].transpose(-12), quarterLength=duration_ql
            )
        elif event_type == "minor_third_low" and scale_pitches:
            elem = note.Note(
                scale_pitches["third"].transpose(-12), quarterLength=duration_ql
            )
        elif event_type == "fifth_low" and scale_pitches:
            elem = note.Note(
                scale_pitches["fifth"].transpose(-12), quarterLength=duration_ql
            )
        elif event_type == "chord_high_voices":
            high_pitches = (
                sorted(pitches, key=lambda p: p.ps)[-3:]
                if len(pitches) >= 3
                else pitches
            )
            elem = m21chord.Chord(high_pitches, quarterLength=duration_ql)
            selected_pitch = min(
                high_pitches, key=lambda p: abs(p.ps - selected_pitch.ps)
            )
        elif event_type == "chord":
            elem = m21chord.Chord(pitches, quarterLength=duration_ql)
        else:
            logger.warning(
                f"Unknown pattern event type: '{event_type}'. Using default for {hand}."
            )
            if hand == "RH":
                elem = m21chord.Chord(pitches, quarterLength=duration_ql)
            else:
                elem = note.Note(selected_pitch, quarterLength=duration_ql)

        if isinstance(elem, m21chord.Chord):
            chosen = min(elem.pitches, key=lambda p: abs(p.ps - selected_pitch.ps))
        else:
            chosen = elem.pitch
        self._prev_last_pitch[hand] = chosen
        return elem

    def _insert_pedal(self, part: stream.Part, section_data: dict[str, Any]):
        intensity = section_data.get("musical_intent", {}).get("intensity", "medium")
        end_offset = section_data.get("q_length", 4.0)
        if not part.flatten().notes:
            return

        cc_events = self._pedal_marks(intensity, end_offset)
        for ev in cc_events:
            if ev["val"] and hasattr(expressions, "PedalType"):
                ped = PedalMark()
                if hasattr(ped, "pedalType"):
                    ped.pedalType = expressions.PedalType.Sustain
                if hasattr(ped, "pedalForm"):
                    ped.pedalForm = expressions.PedalForm.Line
                part.insert(ev["time"], ped)
        part.extra_cc = getattr(part, "extra_cc", []) + cc_events

    def _pedal_marks(self, intensity: str, length_ql: float) -> list[dict[str, Any]]:
        measure_len = self.measure_duration or 4.0
        pedal_value = 127 if intensity == "high" else 64
        events = []
        t = 0.0
        while t < length_ql:
            events.append({"time": t, "cc": 64, "val": pedal_value})
            events.append({"time": min(t + measure_len, length_ql), "cc": 64, "val": 0})
            t += measure_len
        return events

    def _insert_cc11_curves(self, part: stream.Part):
        notes = sorted(part.flatten().notes, key=lambda n: n.offset)
        if len(notes) < 2:
            return
        cc_events = getattr(part, "extra_cc", [])
        for idx, cur in enumerate(notes[:-1]):
            nxt = notes[idx + 1]
            start_val = cur.volume.velocity or 64
            end_val = nxt.volume.velocity or 64
            start_t = cur.offset
            end_t = nxt.offset
            if end_t <= start_t:
                continue
            steps = max(2, int((end_t - start_t) * 4))
            for s in range(steps + 1):
                frac = s / steps
                t = start_t + frac * (end_t - start_t)
                val = int(round(start_val + (end_val - start_val) * frac))
                cc_events.append({"time": t, "cc": 11, "val": max(0, min(127, val))})
        part.extra_cc = cc_events

    def _apply_weak_beat(self, part: stream.Part, style: str) -> stream.Part:
        if style in (None, "none"):
            return part
        beats = (
            self.global_time_signature_obj.beatCount
            if self.global_time_signature_obj
            else 4
        )
        beat_len = (
            self.global_time_signature_obj.beatDuration.quarterLength
            if self.global_time_signature_obj
            else 1.0
        )
        new_part = stream.Part(id=part.id)
        for elem in part.flatten().notesAndRests:
            rel = elem.offset
            beat_pos = rel / beat_len
            is_on_beat = abs(beat_pos - round(beat_pos)) < 0.001
            beat_idx = int(round(beat_pos))
            is_weak = False
            if is_on_beat:
                if beats == 4 and (beat_idx == 1 or beat_idx == 3):
                    is_weak = True
                elif beats == 3 and (beat_idx == 1 or beat_idx == 2):
                    is_weak = True
            if is_weak:
                if style == "rest":
                    continue
                elif style == "ghost":
                    base_vel = (
                        elem.volume.velocity
                        if elem.volume and elem.volume.velocity
                        else 64
                    )
                    elem.volume = m21volume.Volume(velocity=max(1, int(base_vel * 0.4)))
            new_part.insert(elem.offset, elem)
        return new_part

    def _add_fill(
        self, part: stream.Part, cs: harmony.ChordSymbol, length_beats: float
    ) -> stream.Part:
        if length_beats <= 0:
            return part
        offset = self.measure_duration - length_beats
        if offset < 0:
            return part
        fill_chord = m21chord.Chord(cs.pitches, quarterLength=length_beats)
        fill_chord.volume = m21volume.Volume(velocity=80)
        part.insert(offset, fill_chord)
        return part

    def _add_mordent(self, part: stream.Part) -> None:
        """Attach a Mordent expression to the first note in the part."""
        first_note = next((n for n in part.recurse().notes), None)
        if first_note:
            first_note.expressions.append(expressions.Mordent())

    def _add_grace_note(self, part: stream.Part, cs: harmony.ChordSymbol) -> None:
        """Insert a simple grace note a second above the first note."""
        first_note = next((n for n in part.recurse().notes), None)
        if not first_note:
            return
        grace = first_note.getGrace(inPlace=False)
        try:
            grace.pitch = cs.root().transpose(2)
        except Exception:
            grace.pitch = first_note.pitch
        part.insert(first_note.offset, grace)

    def _apply_measure_rubato(
        self, part: stream.Part, amplitude_sec: float = 0.02
    ) -> None:
        """Randomly shift all notes per measure within +/- amplitude_sec."""
        if not part.recurse().notes or not self.global_tempo:
            return
        sec_to_ql = self.global_tempo / 60.0
        shift_amt = amplitude_sec * sec_to_ql
        mlen = self.measure_duration or 4.0
        shifts: dict[int, float] = {}
        for n in part.recurse().notes:
            idx = int(n.offset // mlen)
            if idx not in shifts:
                shifts[idx] = self.rng.uniform(-shift_amt, shift_amt)
            n.offset = max(0.0, n.offset + shifts[idx])
        for cc in getattr(part, "extra_cc", []):
            idx = int(cc["time"] // mlen)
            if idx in shifts:
                cc["time"] = max(0.0, cc["time"] + shifts[idx])

    def _render_part(
        self,
        section_data: dict[str, Any],
        next_section_data: dict[str, Any] | None = None,
        vocal_metrics: dict | None = None,
    ) -> dict[str, stream.Part]:
        rh_part = stream.Part(id=f"Piano_RH_{section_data.get('absolute_offset')}")
        lh_part = stream.Part(id=f"Piano_LH_{section_data.get('absolute_offset')}")
        chord_label = section_data.get("chord_symbol_for_voicing", "Rest")

        if not chord_label or chord_label == "Rest":
            dur = section_data.get("q_length", 4.0)
            rh_part.insert(0, note.Rest(quarterLength=dur))
            lh_part.insert(0, note.Rest(quarterLength=dur))
            rh_part.id = "piano_rh"
            rh_part.partName = "Piano RH"
            lh_part.id = "piano_lh"
            lh_part.partName = "Piano LH"
            return {"piano_rh": rh_part, "piano_lh": lh_part}
        try:
            cs = harmony.ChordSymbol(chord_label)
        except Exception as e:
            logger.error(f"Failed to parse chord '{chord_label}': {e}. Inserting Rest.")
            dur = section_data.get("q_length", 4.0)
            rh_part.insert(0, note.Rest(quarterLength=dur))
            lh_part.insert(0, note.Rest(quarterLength=dur))
            rh_part.id = "piano_rh"
            rh_part.partName = "Piano RH"
            lh_part.id = "piano_lh"
            lh_part.partName = "Piano LH"
            return {"piano_rh": rh_part, "piano_lh": lh_part}

        duration_ql = section_data.get("q_length", 4.0)
        musical_intent = section_data.get("musical_intent", {})
        piano_params = section_data.get("part_params", {}).get("piano", {})

        rh_key = None
        lh_key = None
        if self.overrides:
            rh_key = getattr(self.overrides, "rhythm_key_rh", None) or getattr(
                self.overrides, "rhythm_key", None
            )
            lh_key = getattr(self.overrides, "rhythm_key_lh", None) or getattr(
                self.overrides, "rhythm_key", None
            )

        if not rh_key:
            rh_key = piano_params.get("rhythm_key_rh") or piano_params.get("rhythm_key")
        if not lh_key:
            lh_key = piano_params.get("rhythm_key_lh") or piano_params.get("rhythm_key")

        if not rh_key or not lh_key:
            def_rh, def_lh = self._get_pattern_keys(musical_intent, None)
            rh_key = rh_key or def_rh
            lh_key = lh_key or def_lh

        if rh_key not in self.part_parameters:
            cand = self._find_pattern_by_tags(rh_key.split(), "rh")
            if cand:
                rh_key = cand
        if lh_key not in self.part_parameters:
            cand = self._find_pattern_by_tags(lh_key.split(), "lh")
            if cand:
                lh_key = cand

        section_name = section_data.get("section_name")
        if section_name != self._current_cycle_section:
            self._style_cycle_index = {"RH": 0, "LH": 0}
            self._current_cycle_section = section_name
        style_sequence = piano_params.get(
            "voicing_style_sequence", ["spread", "closed", "inverted"]
        )

        rh_style = style_sequence[self._style_cycle_index["RH"] % len(style_sequence)]
        lh_style = style_sequence[self._style_cycle_index["LH"] % len(style_sequence)]
        self._style_cycle_index["RH"] += 1
        self._style_cycle_index["LH"] += 1

        mode = section_data.get("mode", self.global_key_signature_mode)
        ov = self.overrides.model_dump(exclude_unset=True) if self.overrides else {}
        rh_part = self._render_hand_part(
            "RH",
            cs,
            duration_ql,
            rh_key,
            piano_params,
            voicing_style=rh_style,
            mode=mode,
            velocity_shift=ov.get("velocity_shift_rh"),
        )
        lh_part = self._render_hand_part(
            "LH",
            cs,
            duration_ql,
            lh_key,
            piano_params,
            voicing_style=lh_style,
            mode=mode,
            velocity_shift=ov.get("velocity_shift_lh"),
        )

        rh_part = self._apply_weak_beat(rh_part, ov.get("weak_beat_style_rh", "none"))
        lh_part = self._apply_weak_beat(lh_part, ov.get("weak_beat_style_lh", "none"))

        for part in (rh_part, lh_part):
            self._apply_measure_rubato(part)

        if ov.get("fill_on_4th"):
            fill_len = float(ov.get("fill_length_beats", 0.5))
            rh_part = self._add_fill(rh_part, cs, fill_len)

        if ov.get("mordent"):
            self._add_mordent(rh_part)

        if ov.get("grace_note"):
            self._add_grace_note(rh_part, cs)

        for element in rh_part.flatten().notesAndRests:
            element.stemDirection = "up"
        for element in lh_part.flatten().notesAndRests:
            element.stemDirection = "down"

        for part in (rh_part, lh_part):
            self._insert_pedal(part, section_data)
            self._insert_cc11_curves(part)
            part.insert(0, self.default_instrument)

        profile_name = (
            self.cfg.get("humanize_profile")
            or section_data.get("humanize_profile")
            or self.global_settings.get("humanize_profile")
        )
        for part in (rh_part, lh_part):
            if profile_name:
                humanizer.apply(
                    part,
                    profile_name,
                    global_settings=self.global_settings,
                )

        global_profile = self.cfg.get(
            "global_humanize_profile"
        ) or self.global_settings.get("global_humanize_profile")
        for part in (rh_part, lh_part):
            if global_profile:
                humanizer.apply(
                    part,
                    global_profile,
                    global_settings=self.global_settings,
                )

        # 結合して 1 Part を返していたコードを削除
        rh_part.id = "piano_rh"
        rh_part.partName = "Piano RH"
        lh_part.id = "piano_lh"
        lh_part.partName = "Piano LH"

        if vocal_metrics:
            root_pitch = cs.root() if "cs" in locals() else None
            if root_pitch:
                if self.anticipatory_chord:
                    for start, end in get_rest_windows(vocal_metrics):
                        off = end - 0.125
                        if off >= start:
                            n = note.Note()
                            n.pitch = root_pitch.transpose(12)
                            n.duration = m21duration.Duration(0.25)
                            n.volume = m21volume.Volume(velocity=75)
                            rh_part.insert(off, n)
                for b in vocal_metrics.get("onsets", []):
                    if random.random() < 0.3:
                        n = note.Note()
                        n.pitch = root_pitch.transpose(12)
                        n.duration = m21duration.Duration(0.25)
                        n.volume = m21volume.Volume(velocity=75)
                        rh_part.insert(b + 0.5, n)

        return {"piano_rh": rh_part, "piano_lh": lh_part}
